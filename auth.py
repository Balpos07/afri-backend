import os
import secrets
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

import config
from db import Admin, get_session

_configured_secret = config.get("JWT_SECRET")
if not _configured_secret:
    print("WARNING: JWT_SECRET is not set; using a random one. Admin sessions end on every restart.")
JWT_SECRET = _configured_secret or secrets.token_urlsafe(48)
JWT_ALGORITHM = "HS256"
TOKEN_MINUTES = int(os.getenv("ADMIN_TOKEN_MINUTES", "60"))

MAX_FAILURES = 5
WINDOW_SECONDS = 15 * 60
_failures = defaultdict(list)

bearer = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def create_token(admin: Admin) -> str:
    payload = {
        "sub": str(admin.id),
        "email": admin.email,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=TOKEN_MINUTES),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def _client_keys(request: Request, email: str):
    ip = request.client.host if request.client else "unknown"
    return (f"ip:{ip}", f"email:{email.lower()}")


def check_rate_limit(request: Request, email: str):
    now = time.time()
    for key in _client_keys(request, email):
        _failures[key] = [t for t in _failures[key] if now - t < WINDOW_SECONDS]
        if len(_failures[key]) >= MAX_FAILURES:
            raise HTTPException(status_code=429, detail="Too many failed attempts. Try again in 15 minutes.")


def register_failure(request: Request, email: str):
    now = time.time()
    for key in _client_keys(request, email):
        _failures[key].append(now)


def clear_failures(request: Request, email: str):
    for key in _client_keys(request, email):
        _failures.pop(key, None)


def get_current_admin(
    creds: HTTPAuthorizationCredentials = Depends(bearer),
    session: Session = Depends(get_session),
) -> Admin:
    if creds is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(creds.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Session expired. Please log in again.")
    admin = session.scalar(select(Admin).where(Admin.id == int(payload["sub"])))
    if admin is None or not admin.is_active:
        raise HTTPException(status_code=401, detail="Account not found or disabled")
    return admin
