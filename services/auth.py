"""
Authentication service — user registration, login, JWT tokens.
SQLite-backed user store with bcrypt password hashing.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional

import bcrypt
import jwt

from app.config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class User:
    __slots__ = ("user_id", "username", "email", "password_hash", "created_at")

    def __init__(self, user_id: str, username: str, email: str, password_hash: str, created_at: str):
        self.user_id = user_id
        self.username = username
        self.email = email
        self.password_hash = password_hash
        self.created_at = created_at

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "created_at": self.created_at,
        }


class AuthStore:
    """SQLite-backed user authentication store."""

    def __init__(self, db_path: str = None) -> None:
        self._db_path = db_path or settings.USERS_DB_PATH
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def register(self, username: str, email: str, password: str) -> Optional[User]:
        """Register a new user. Returns None if username or email already exists."""
        if len(password) < 6:
            raise ValueError("Password must be at least 6 characters")

        password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        user_id = str(uuid.uuid4())[:8]
        now = datetime.now(timezone.utc).isoformat()

        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                "INSERT INTO users VALUES (?, ?, ?, ?, ?)",
                (user_id, username, email, password_hash, now),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            return None
        finally:
            conn.close()

        logger.info(f"Registered user: {username} ({user_id})")
        return User(user_id=user_id, username=username, email=email, password_hash=password_hash, created_at=now)

    def login(self, username: str, password: str) -> Optional[User]:
        """Authenticate a user. Returns None if credentials are invalid."""
        conn = sqlite3.connect(self._db_path)
        try:
            row = conn.execute(
                "SELECT user_id, username, email, password_hash, created_at FROM users WHERE username=?",
                (username,),
            ).fetchone()
        finally:
            conn.close()

        if not row:
            return None

        user = User(user_id=row[0], username=row[1], email=row[2], password_hash=row[3], created_at=row[4])

        if not bcrypt.checkpw(password.encode("utf-8"), user.password_hash.encode("utf-8")):
            return None

        logger.info(f"User logged in: {username}")
        return user

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        conn = sqlite3.connect(self._db_path)
        try:
            row = conn.execute(
                "SELECT user_id, username, email, password_hash, created_at FROM users WHERE user_id=?",
                (user_id,),
            ).fetchone()
        finally:
            conn.close()

        if not row:
            return None
        return User(user_id=row[0], username=row[1], email=row[2], password_hash=row[3], created_at=row[4])

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        conn = sqlite3.connect(self._db_path)
        try:
            row = conn.execute(
                "SELECT user_id, username, email, password_hash, created_at FROM users WHERE username=?",
                (username,),
            ).fetchone()
        finally:
            conn.close()

        if not row:
            return None
        return User(user_id=row[0], username=row[1], email=row[2], password_hash=row[3], created_at=row[4])

    @staticmethod
    def create_token(user: User) -> str:
        """Create a JWT token for a user."""
        payload = {
            "sub": user.user_id,
            "username": user.username,
            "exp": datetime.now(timezone.utc) + timedelta(hours=settings.JWT_EXPIRY_HOURS),
            "iat": datetime.now(timezone.utc),
        }
        return jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")

    @staticmethod
    def decode_token(token: str) -> Optional[dict]:
        """Decode and verify a JWT token. Returns payload dict or None."""
        try:
            payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
            return payload
        except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
            return None
