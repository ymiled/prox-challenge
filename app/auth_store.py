from __future__ import annotations

import base64
import hashlib
import secrets
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from cryptography.fernet import Fernet

try:
    import psycopg
    from psycopg.errors import UniqueViolation
    from psycopg.rows import dict_row
except ImportError:  # pragma: no cover - optional until Postgres is configured
    psycopg = None
    UniqueViolation = None
    dict_row = None


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class UserRecord:
    id: int
    username: str
    created_at: str


@dataclass
class SessionRecord:
    token: str
    csrf_token: str
    expires_at: str


class DuplicateUserError(Exception):
    pass


class AuthStore:
    def __init__(self, database_url: str | None, db_path: Path, secret: str) -> None:
        self.database_url = (database_url or "").strip()
        self.db_path = db_path
        self.uses_postgres = self.database_url.startswith(("postgres://", "postgresql://"))
        if not self.uses_postgres:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        elif psycopg is None:
            raise RuntimeError("psycopg is required when DATABASE_URL points to Postgres.")

        derived = hashlib.sha256(secret.encode("utf-8")).digest()
        self.cipher = Fernet(base64.urlsafe_b64encode(derived))
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[Any]:
        if self.uses_postgres:
            conn = psycopg.connect(self.database_url, autocommit=False, row_factory=dict_row)
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
            return

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _placeholder(self, index: int) -> str:
        return "%s" if self.uses_postgres else "?"

    def _placeholders(self, count: int) -> str:
        return ", ".join(self._placeholder(index) for index in range(count))

    def _row_value(self, row: Any, key: str) -> Any:
        if row is None:
            return None
        return row[key]

    def _init_db(self) -> None:
        with self._connect() as conn:
            cursor = conn.cursor()
            if self.uses_postgres:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        id BIGSERIAL PRIMARY KEY,
                        email TEXT NOT NULL UNIQUE,
                        password_salt TEXT NOT NULL,
                        password_hash TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        token_hash TEXT PRIMARY KEY,
                        user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                        created_at TEXT NOT NULL,
                        expires_at TEXT NOT NULL,
                        csrf_token_hash TEXT
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS api_credentials (
                        user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                        provider TEXT NOT NULL,
                        encrypted_api_key TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY(user_id, provider)
                    )
                    """
                )
                cursor.execute(
                    """
                    ALTER TABLE sessions
                    ADD COLUMN IF NOT EXISTS csrf_token_hash TEXT
                    """
                )
                return

            cursor.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT NOT NULL UNIQUE,
                    password_salt TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS api_credentials (
                    user_id INTEGER NOT NULL,
                    provider TEXT NOT NULL,
                    encrypted_api_key TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(user_id, provider),
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );
                """
            )
            session_columns = {row["name"] for row in cursor.execute("PRAGMA table_info(sessions)").fetchall()}
            if "csrf_token_hash" not in session_columns:
                cursor.execute("ALTER TABLE sessions ADD COLUMN csrf_token_hash TEXT")

    def _hash_password(self, password: str, salt: str) -> str:
        return hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt),
            200_000,
        ).hex()

    def create_user(self, username: str, password: str) -> UserRecord:
        normalized = username.strip().lower()
        salt = secrets.token_hex(16)
        password_hash = self._hash_password(password, salt)
        created_at = utcnow().isoformat()
        with self._connect() as conn:
            cursor = conn.cursor()
            try:
                if self.uses_postgres:
                    cursor.execute(
                        """
                        INSERT INTO users(email, password_salt, password_hash, created_at)
                        VALUES(%s, %s, %s, %s)
                        RETURNING id
                        """,
                        (normalized, salt, password_hash, created_at),
                    )
                    row = cursor.fetchone()
                    user_id = int(self._row_value(row, "id"))
                else:
                    cursor.execute(
                        "INSERT INTO users(email, password_salt, password_hash, created_at) VALUES(?, ?, ?, ?)",
                        (normalized, salt, password_hash, created_at),
                    )
                    user_id = int(cursor.lastrowid)
            except Exception as exc:
                if self._is_duplicate_user_error(exc):
                    raise DuplicateUserError from exc
                raise
        return UserRecord(id=user_id, username=normalized, created_at=created_at)

    def _is_duplicate_user_error(self, exc: Exception) -> bool:
        if isinstance(exc, sqlite3.IntegrityError):
            return True
        if UniqueViolation is not None and isinstance(exc, UniqueViolation):
            return True
        cause = getattr(exc, "__cause__", None)
        return bool(UniqueViolation is not None and isinstance(cause, UniqueViolation))

    def authenticate_user(self, username: str, password: str) -> UserRecord | None:
        normalized = username.strip().lower()
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT id, email, created_at, password_salt, password_hash
                FROM users
                WHERE email = {self._placeholder(1)}
                """,
                (normalized,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        candidate = self._hash_password(password, self._row_value(row, "password_salt"))
        if not secrets.compare_digest(candidate, self._row_value(row, "password_hash")):
            return None
        return UserRecord(
            id=int(self._row_value(row, "id")),
            username=str(self._row_value(row, "email")),
            created_at=str(self._row_value(row, "created_at")),
        )

    def get_user(self, user_id: int) -> UserRecord | None:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT id, email, created_at FROM users WHERE id = {self._placeholder(1)}",
                (user_id,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return UserRecord(
            id=int(self._row_value(row, "id")),
            username=str(self._row_value(row, "email")),
            created_at=str(self._row_value(row, "created_at")),
        )

    def create_session(self, user_id: int, ttl_days: int = 30) -> SessionRecord:
        token = secrets.token_urlsafe(32)
        csrf_token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        csrf_token_hash = hashlib.sha256(csrf_token.encode("utf-8")).hexdigest()
        created_at = utcnow()
        expires_at = created_at + timedelta(days=ttl_days)
        with self._connect() as conn:
            cursor = conn.cursor()
            if self.uses_postgres:
                cursor.execute(
                    """
                    INSERT INTO sessions(token_hash, user_id, created_at, expires_at, csrf_token_hash)
                    VALUES(%s, %s, %s, %s, %s)
                    ON CONFLICT (token_hash) DO UPDATE SET
                        user_id = EXCLUDED.user_id,
                        created_at = EXCLUDED.created_at,
                        expires_at = EXCLUDED.expires_at,
                        csrf_token_hash = EXCLUDED.csrf_token_hash
                    """,
                    (token_hash, user_id, created_at.isoformat(), expires_at.isoformat(), csrf_token_hash),
                )
            else:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO sessions(token_hash, user_id, created_at, expires_at, csrf_token_hash)
                    VALUES(?, ?, ?, ?, ?)
                    """,
                    (token_hash, user_id, created_at.isoformat(), expires_at.isoformat(), csrf_token_hash),
                )
        return SessionRecord(token=token, csrf_token=csrf_token, expires_at=expires_at.isoformat())

    def get_user_by_session(self, token: str | None) -> UserRecord | None:
        if not token:
            return None
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT users.id, users.email, users.created_at, sessions.expires_at
                FROM sessions
                JOIN users ON users.id = sessions.user_id
                WHERE sessions.token_hash = {self._placeholder(1)}
                """,
                (token_hash,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            expires_at = datetime.fromisoformat(str(self._row_value(row, "expires_at")))
            if expires_at <= utcnow():
                cursor.execute(
                    f"DELETE FROM sessions WHERE token_hash = {self._placeholder(1)}",
                    (token_hash,),
                )
                return None
        return UserRecord(
            id=int(self._row_value(row, "id")),
            username=str(self._row_value(row, "email")),
            created_at=str(self._row_value(row, "created_at")),
        )

    def delete_session(self, token: str | None) -> None:
        if not token:
            return
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with self._connect() as conn:
            conn.cursor().execute(
                f"DELETE FROM sessions WHERE token_hash = {self._placeholder(1)}",
                (token_hash,),
            )

    def validate_session_csrf(self, token: str | None, csrf_token: str | None) -> bool:
        if not token or not csrf_token:
            return False
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        csrf_hash = hashlib.sha256(csrf_token.encode("utf-8")).hexdigest()
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT csrf_token_hash, expires_at
                FROM sessions
                WHERE token_hash = {self._placeholder(1)}
                """,
                (token_hash,),
            )
            row = cursor.fetchone()
            if row is None:
                return False
            expires_at = datetime.fromisoformat(str(self._row_value(row, "expires_at")))
            if expires_at <= utcnow():
                cursor.execute(
                    f"DELETE FROM sessions WHERE token_hash = {self._placeholder(1)}",
                    (token_hash,),
                )
                return False
        stored_hash = str(self._row_value(row, "csrf_token_hash") or "")
        return secrets.compare_digest(stored_hash, csrf_hash)

    def delete_expired_sessions(self) -> int:
        timestamp = utcnow().isoformat()
        with self._connect() as conn:
            cursor = conn.cursor()
            if self.uses_postgres:
                cursor.execute(
                    "DELETE FROM sessions WHERE expires_at <= %s",
                    (timestamp,),
                )
            else:
                cursor.execute(
                    "DELETE FROM sessions WHERE expires_at <= ?",
                    (timestamp,),
                )
            return int(cursor.rowcount or 0)

    def upsert_api_key(self, user_id: int, provider: str, api_key: str) -> None:
        encrypted = self.cipher.encrypt(api_key.encode("utf-8")).decode("utf-8")
        timestamp = utcnow().isoformat()
        with self._connect() as conn:
            cursor = conn.cursor()
            if self.uses_postgres:
                cursor.execute(
                    """
                    INSERT INTO api_credentials(user_id, provider, encrypted_api_key, created_at, updated_at)
                    VALUES(%s, %s, %s, %s, %s)
                    ON CONFLICT(user_id, provider) DO UPDATE SET
                        encrypted_api_key = EXCLUDED.encrypted_api_key,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (user_id, provider, encrypted, timestamp, timestamp),
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO api_credentials(user_id, provider, encrypted_api_key, created_at, updated_at)
                    VALUES(?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, provider) DO UPDATE SET
                        encrypted_api_key = excluded.encrypted_api_key,
                        updated_at = excluded.updated_at
                    """,
                    (user_id, provider, encrypted, timestamp, timestamp),
                )

    def delete_api_key(self, user_id: int, provider: str) -> None:
        with self._connect() as conn:
            conn.cursor().execute(
                f"""
                DELETE FROM api_credentials
                WHERE user_id = {self._placeholder(1)} AND provider = {self._placeholder(2)}
                """,
                (user_id, provider),
            )

    def get_api_key(self, user_id: int, provider: str) -> str:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT encrypted_api_key
                FROM api_credentials
                WHERE user_id = {self._placeholder(1)} AND provider = {self._placeholder(2)}
                """,
                (user_id, provider),
            )
            row = cursor.fetchone()
        if row is None:
            return ""
        encrypted = str(self._row_value(row, "encrypted_api_key"))
        return self.cipher.decrypt(encrypted.encode("utf-8")).decode("utf-8")

    def has_api_key(self, user_id: int, provider: str) -> bool:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT 1
                FROM api_credentials
                WHERE user_id = {self._placeholder(1)} AND provider = {self._placeholder(2)}
                """,
                (user_id, provider),
            )
            row = cursor.fetchone()
        return row is not None
