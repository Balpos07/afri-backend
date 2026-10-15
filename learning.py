"""Progressive-learning logic: capture user inputs, queue them for review,
and apply admin decisions to the live knowledge base."""
import json
import re
from pathlib import Path

from sqlalchemy import select

from db import Candidate, KBAudit, KBEntry, SessionLocal, UserInput, utcnow

KB_JSON = Path(__file__).parent / "knowledge_base.json"

# Fuzzy matches below this are queued for review even though they got an answer
BORDERLINE_SCORE = 0.75

# Set by main.py so an approval can refresh the in-memory matcher
kb_changed_hook = None

_PII_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[email]"),
    (re.compile(r"\+?\d[\d\s-]{6,}\d"), "[number]"),
]


def scrub(text: str) -> str:
    """Remove emails and phone-like numbers before anything is stored."""
    for pattern, repl in _PII_PATTERNS:
        text = pattern.sub(repl, text)
    return text.strip()


def normalize(text: str) -> str:
    text = re.sub(r"[^\w\s]", "", text.lower())
    return re.sub(r"\s+", " ", text).strip()[:255]


def entry_snapshot(entry: KBEntry) -> dict:
    return {
        "symptom": entry.symptom,
        "phrases": list(entry.phrases or []),
        "recommendation": entry.recommendation,
        "active": entry.active,
    }


# ── Knowledge base loading ──

def _load_json_kb() -> list:
    with open(KB_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def seed_kb():
    """Copy knowledge_base.json into kb_entries the first time the table is empty."""
    with SessionLocal() as session:
        if session.scalar(select(KBEntry.id).limit(1)) is not None:
            return 0
        items = _load_json_kb()
        for item in items:
            phrases = item["pidgin"] if isinstance(item["pidgin"], list) else [item["pidgin"]]
            session.add(
                KBEntry(
                    symptom=item["symptom"],
                    phrases=phrases,
                    recommendation=item["recommendation"],
                )
            )
        session.commit()
        return len(items)


def load_kb() -> list:
    """Active entries in the shape the matcher expects. Falls back to the JSON file."""
    try:
        with SessionLocal() as session:
            rows = session.scalars(select(KBEntry).where(KBEntry.active.is_(True))).all()
            if rows:
                return [
                    {"symptom": r.symptom, "pidgin": list(r.phrases or []), "recommendation": r.recommendation}
                    for r in rows
                ]
    except Exception as e:
        print(f"WARNING: could not load KB from database ({e}); using knowledge_base.json")
    return _load_json_kb()


def notify_kb_changed():
    if kb_changed_hook:
        try:
            kb_changed_hook()
        except Exception as e:
            print(f"WARNING: KB reload failed: {e}")


# ── Capture ──

def _upsert_candidate(session, text, score, symptom, phrase, reason):
    key = normalize(text)
    if not key:
        return
    cand = session.scalar(select(Candidate).where(Candidate.text_key == key))
    if cand is None:
        session.add(
            Candidate(
                text_key=key,
                example_text=scrub(text)[:1000],
                best_score=score or 0.0,
                suggested_symptom=symptom,
                suggested_phrase=phrase,
                reason=reason,
            )
        )
        return
    cand.count += 1
    cand.last_seen = utcnow()
    if (score or 0.0) > cand.best_score:
        cand.best_score = score
        cand.suggested_symptom = symptom
        cand.suggested_phrase = phrase
    # A rejected/approved text that keeps coming back is worth another look
    if cand.status == "rejected" and cand.count % 5 == 0:
        cand.status = "pending"


def record(text, source, outcome, score=None, symptom=None, phrase=None):
    """Log a query. Never raises, so a database problem can't break the assistant."""
    try:
        clean = scrub(text or "")
        if not clean:
            return
        with SessionLocal() as session:
            session.add(
                UserInput(text=clean[:1000], source=source, outcome=outcome, score=score, matched_symptom=symptom)
            )
            if outcome == "miss":
                _upsert_candidate(session, text, score, symptom, phrase, "miss")
            elif outcome == "fuzzy" and (score or 0.0) < BORDERLINE_SCORE:
                _upsert_candidate(session, text, score, symptom, phrase, "low_confidence")
            session.commit()
    except Exception as e:
        print(f"WARNING: could not record user input: {e}")


def record_feedback(text, helpful: bool, symptom=None):
    try:
        clean = scrub(text or "")
        if not clean:
            return
        with SessionLocal() as session:
            session.add(
                UserInput(
                    text=clean[:1000],
                    source="feedback",
                    outcome="helpful" if helpful else "unhelpful",
                    matched_symptom=symptom,
                )
            )
            if not helpful:
                _upsert_candidate(session, text, 0.0, symptom, None, "unhelpful")
            session.commit()
    except Exception as e:
        print(f"WARNING: could not record feedback: {e}")


# ── Admin decisions ──

class ReviewError(Exception):
    pass


def _audit(session, entry_id, action, before, after, admin_email, candidate_id=None):
    session.add(
        KBAudit(
            entry_id=entry_id,
            action=action,
            before=before,
            after=after,
            admin_email=admin_email,
            candidate_id=candidate_id,
        )
    )


def approve_candidate(session, cand: Candidate, admin_email, mode, phrase, recommendation=None,
                      symptom=None, entry_id=None, note=None):
    """mode='existing': add `phrase` to an existing entry.
    mode='new': create a new entry with `symptom`, `phrase` and `recommendation`."""
    phrase = normalize(phrase or cand.example_text)
    if not phrase:
        raise ReviewError("A trigger phrase is required.")

    if mode == "existing":
        entry = session.get(KBEntry, entry_id) if entry_id else None
        if entry is None:
            raise ReviewError("Choose an existing knowledge-base entry.")
        before = entry_snapshot(entry)
        if phrase not in [p.lower() for p in entry.phrases]:
            entry.phrases = list(entry.phrases) + [phrase]
        entry.version += 1
        entry.updated_at = utcnow()
        _audit(session, entry.id, "add_phrase", before, entry_snapshot(entry), admin_email, cand.id)
    elif mode == "new":
        symptom = re.sub(r"\s+", "_", (symptom or "").strip().lower())
        if not symptom or not (recommendation or "").strip():
            raise ReviewError("A symptom name and a recommendation are required.")
        if session.scalar(select(KBEntry.id).where(KBEntry.symptom == symptom)):
            raise ReviewError(f"Symptom '{symptom}' already exists; add the phrase to it instead.")
        entry = KBEntry(symptom=symptom, phrases=[phrase], recommendation=recommendation.strip())
        session.add(entry)
        session.flush()
        _audit(session, entry.id, "create", None, entry_snapshot(entry), admin_email, cand.id)
    else:
        raise ReviewError("mode must be 'new' or 'existing'.")

    cand.status = "approved"
    cand.reviewed_by = admin_email
    cand.reviewed_at = utcnow()
    cand.note = note
    session.commit()
    notify_kb_changed()
    return entry


def reject_candidate(session, cand: Candidate, admin_email, note=None):
    cand.status = "rejected"
    cand.reviewed_by = admin_email
    cand.reviewed_at = utcnow()
    cand.note = note
    session.commit()


def update_entry(session, entry: KBEntry, admin_email, symptom=None, phrases=None,
                 recommendation=None, active=None):
    before = entry_snapshot(entry)
    if symptom is not None:
        symptom = re.sub(r"\s+", "_", symptom.strip().lower())
        clash = session.scalar(select(KBEntry.id).where(KBEntry.symptom == symptom, KBEntry.id != entry.id))
        if not symptom or clash:
            raise ReviewError("Symptom name is empty or already used by another entry.")
        entry.symptom = symptom
    if phrases is not None:
        cleaned = [p for p in (normalize(x) for x in phrases) if p]
        if not cleaned:
            raise ReviewError("At least one trigger phrase is required.")
        entry.phrases = cleaned
    if recommendation is not None:
        if not recommendation.strip():
            raise ReviewError("Recommendation cannot be empty.")
        entry.recommendation = recommendation.strip()
    if active is not None:
        entry.active = active
    entry.version += 1
    entry.updated_at = utcnow()
    _audit(session, entry.id, "update", before, entry_snapshot(entry), admin_email)
    session.commit()
    notify_kb_changed()


def revert_audit(session, audit: KBAudit, admin_email):
    """Undo one change: restore its 'before' snapshot (or deactivate if it was a create)."""
    entry = session.get(KBEntry, audit.entry_id) if audit.entry_id else None
    if entry is None:
        raise ReviewError("The entry for this change no longer exists.")
    current = entry_snapshot(entry)
    if audit.before is None:
        entry.active = False
    else:
        entry.symptom = audit.before["symptom"]
        entry.phrases = audit.before["phrases"]
        entry.recommendation = audit.before["recommendation"]
        entry.active = audit.before["active"]
    entry.version += 1
    entry.updated_at = utcnow()
    _audit(session, entry.id, "revert", current, entry_snapshot(entry), admin_email)
    session.commit()
    notify_kb_changed()
