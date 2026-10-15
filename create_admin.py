"""Create or update an admin account. Only a bcrypt hash is stored.

Usage:
    python create_admin.py                       # prompts for email and password
    ADMIN_EMAIL=... ADMIN_PASSWORD=... python create_admin.py

The target database is taken from the DATABASE_URL environment variable
(falls back to a local dev.db SQLite file).
"""
import getpass
import os
import sys

import bcrypt
from sqlalchemy import select

from db import Admin, SessionLocal, engine, init_db


def main():
    email = (os.getenv("ADMIN_EMAIL") or input("Admin email: ")).strip().lower()
    password = os.getenv("ADMIN_PASSWORD") or getpass.getpass("Password: ")

    if not email or not password:
        sys.exit("Email and password are required.")

    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    init_db()
    with SessionLocal() as session:
        admin = session.scalar(select(Admin).where(Admin.email == email))
        if admin:
            admin.password_hash = password_hash
            admin.is_active = True
            action = "updated"
        else:
            session.add(Admin(email=email, password_hash=password_hash, role="admin"))
            action = "created"
        session.commit()

    print(f"Admin {email} {action} on {engine.url.render_as_string(hide_password=True)}")


if __name__ == "__main__":
    main()
