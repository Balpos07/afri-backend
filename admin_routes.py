from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

import learning
from auth import (
    check_rate_limit,
    clear_failures,
    create_token,
    get_current_admin,
    hash_password,
    register_failure,
    verify_password,
)
from db import Admin, Candidate, KBAudit, KBEntry, UserInput, get_session, utcnow

router = APIRouter(prefix="/admin", tags=["admin"])


# ── Schemas ──

class LoginRequest(BaseModel):
    email: str
    password: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


class ApproveRequest(BaseModel):
    mode: str  # "new" | "existing"
    phrase: Optional[str] = None
    entry_id: Optional[int] = None
    symptom: Optional[str] = None
    recommendation: Optional[str] = None
    note: Optional[str] = None


class RejectRequest(BaseModel):
    note: Optional[str] = None


class EntryUpdate(BaseModel):
    symptom: Optional[str] = None
    phrases: Optional[list] = None
    recommendation: Optional[str] = None
    active: Optional[bool] = None


def iso(dt):
    return dt.isoformat() if dt else None


def candidate_dict(c: Candidate):
    return {
        "id": c.id,
        "text": c.example_text,
        "count": c.count,
        "score": round(c.best_score or 0.0, 3),
        "suggested_symptom": c.suggested_symptom,
        "suggested_phrase": c.suggested_phrase,
        "reason": c.reason,
        "status": c.status,
        "created_at": iso(c.created_at),
        "last_seen": iso(c.last_seen),
        "reviewed_by": c.reviewed_by,
        "reviewed_at": iso(c.reviewed_at),
        "note": c.note,
    }


def entry_dict(e: KBEntry):
    return {
        "id": e.id,
        "symptom": e.symptom,
        "phrases": e.phrases or [],
        "recommendation": e.recommendation,
        "version": e.version,
        "active": e.active,
        "updated_at": iso(e.updated_at),
    }


# ── Auth ──

@router.post("/login")
def login(body: LoginRequest, request: Request, session: Session = Depends(get_session)):
    email = body.email.strip().lower()
    check_rate_limit(request, email)

    admin = session.scalar(select(Admin).where(Admin.email == email))
    if admin is None or not admin.is_active or not verify_password(body.password, admin.password_hash):
        register_failure(request, email)
        raise HTTPException(status_code=401, detail="Invalid email or password")

    clear_failures(request, email)
    admin.last_login = utcnow()
    session.commit()
    return {
        "token": create_token(admin),
        "admin": {"email": admin.email, "name": admin.name, "role": admin.role,
                  "must_change_password": admin.must_change_password},
    }


@router.get("/me")
def me(admin: Admin = Depends(get_current_admin)):
    return {"email": admin.email, "name": admin.name, "role": admin.role,
            "must_change_password": admin.must_change_password}


@router.post("/change-password")
def change_password(body: ChangePasswordRequest, admin: Admin = Depends(get_current_admin),
                    session: Session = Depends(get_session)):
    if not verify_password(body.current_password, admin.password_hash):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    if len(body.new_password) < 8:
        raise HTTPException(status_code=400, detail="New password must be at least 8 characters")
    admin = session.get(Admin, admin.id)
    admin.password_hash = hash_password(body.new_password)
    admin.must_change_password = False
    session.commit()
    return {"ok": True}


# ── Overview ──

@router.get("/stats")
def stats(admin: Admin = Depends(get_current_admin), session: Session = Depends(get_session)):
    outcomes = dict(session.execute(
        select(UserInput.outcome, func.count()).where(UserInput.source != "feedback").group_by(UserInput.outcome)
    ).all())
    total = sum(outcomes.values())
    pending = session.scalar(select(func.count()).select_from(Candidate).where(Candidate.status == "pending"))
    return {
        "total_queries": total,
        "exact": outcomes.get("exact", 0),
        "fuzzy": outcomes.get("fuzzy", 0),
        "miss": outcomes.get("miss", 0),
        "miss_rate": round(outcomes.get("miss", 0) / total, 3) if total else 0,
        "pending_candidates": pending,
        "kb_entries": session.scalar(select(func.count()).select_from(KBEntry).where(KBEntry.active.is_(True))),
        "approved": session.scalar(select(func.count()).select_from(Candidate).where(Candidate.status == "approved")),
        "unhelpful_feedback": session.scalar(
            select(func.count()).select_from(UserInput).where(UserInput.outcome == "unhelpful")),
    }


# ── Review queue ──

@router.get("/candidates")
def list_candidates(status: str = "pending", admin: Admin = Depends(get_current_admin),
                    session: Session = Depends(get_session)):
    rows = session.scalars(
        select(Candidate).where(Candidate.status == status)
        .order_by(Candidate.count.desc(), Candidate.last_seen.desc()).limit(200)
    ).all()
    return [candidate_dict(c) for c in rows]


def _get_candidate(session, candidate_id):
    cand = session.get(Candidate, candidate_id)
    if cand is None:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return cand


@router.post("/candidates/{candidate_id}/approve")
def approve(candidate_id: int, body: ApproveRequest, admin: Admin = Depends(get_current_admin),
            session: Session = Depends(get_session)):
    cand = _get_candidate(session, candidate_id)
    try:
        entry = learning.approve_candidate(
            session, cand, admin.email, body.mode, body.phrase,
            recommendation=body.recommendation, symptom=body.symptom,
            entry_id=body.entry_id, note=body.note,
        )
    except learning.ReviewError as e:
        session.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    return {"candidate": candidate_dict(cand), "entry": entry_dict(entry)}


@router.post("/candidates/{candidate_id}/reject")
def reject(candidate_id: int, body: RejectRequest, admin: Admin = Depends(get_current_admin),
           session: Session = Depends(get_session)):
    cand = _get_candidate(session, candidate_id)
    learning.reject_candidate(session, cand, admin.email, body.note)
    return candidate_dict(cand)


# ── Knowledge base ──

@router.get("/kb")
def list_kb(q: str = "", admin: Admin = Depends(get_current_admin), session: Session = Depends(get_session)):
    stmt = select(KBEntry).order_by(KBEntry.symptom)
    rows = session.scalars(stmt).all()
    q = q.strip().lower()
    if q:
        rows = [r for r in rows if q in r.symptom.lower() or any(q in p.lower() for p in (r.phrases or []))]
    return [entry_dict(r) for r in rows]


@router.put("/kb/{entry_id}")
def edit_kb(entry_id: int, body: EntryUpdate, admin: Admin = Depends(get_current_admin),
            session: Session = Depends(get_session)):
    entry = session.get(KBEntry, entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Entry not found")
    try:
        learning.update_entry(session, entry, admin.email, body.symptom, body.phrases,
                              body.recommendation, body.active)
    except learning.ReviewError as e:
        session.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    return entry_dict(entry)


# ── Audit ──

@router.get("/audit")
def audit_log(admin: Admin = Depends(get_current_admin), session: Session = Depends(get_session)):
    rows = session.scalars(select(KBAudit).order_by(KBAudit.created_at.desc()).limit(100)).all()
    return [
        {
            "id": a.id, "entry_id": a.entry_id, "action": a.action,
            "symptom": (a.after or a.before or {}).get("symptom"),
            "before": a.before, "after": a.after,
            "admin": a.admin_email, "candidate_id": a.candidate_id,
            "created_at": iso(a.created_at),
        }
        for a in rows
    ]


@router.post("/audit/{audit_id}/revert")
def revert(audit_id: int, admin: Admin = Depends(get_current_admin), session: Session = Depends(get_session)):
    audit = session.get(KBAudit, audit_id)
    if audit is None:
        raise HTTPException(status_code=404, detail="Audit record not found")
    try:
        learning.revert_audit(session, audit, admin.email)
    except learning.ReviewError as e:
        session.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True}
