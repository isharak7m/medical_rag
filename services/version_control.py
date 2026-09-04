"""
Version control for research artifacts.
Stores notes, hypotheses, and manuscripts with Git-like history.
Each save creates a versioned snapshot with diff tracking.
"""

from __future__ import annotations

import difflib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field

from utils.logger import get_logger

logger = get_logger(__name__)


class ArtifactVersion(BaseModel):
    version_id: str
    artifact_id: str
    content: str
    message: str
    created_at: str
    author: str = "system"
    diff_from_previous: str = ""
    tags: List[str] = Field(default_factory=list)


class ResearchArtifact(BaseModel):
    artifact_id: str
    artifact_type: str  # note, hypothesis, manuscript, review
    title: str
    content: str
    current_version: int = 1
    created_at: str
    updated_at: str
    author: str = "system"
    owner_id: str = ""
    tags: List[str] = Field(default_factory=list)
    versions: List[ArtifactVersion] = Field(default_factory=list)


class VersionControlStore:
    """
    SQLite-backed version control for research artifacts.
    Supports create, update with diff, history, and tag retrieval.
    """

    def __init__(self, db_path: str = "research_artifacts.db") -> None:
        self._db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    artifact_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    current_version INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    author TEXT DEFAULT 'system',
                    tags TEXT DEFAULT '[]',
                    owner_id TEXT DEFAULT ''
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS versions (
                    version_id TEXT PRIMARY KEY,
                    artifact_id TEXT NOT NULL,
                    version_number INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    author TEXT DEFAULT 'system',
                    diff_from_previous TEXT DEFAULT '',
                    tags TEXT DEFAULT '[]',
                    FOREIGN KEY (artifact_id) REFERENCES artifacts(artifact_id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS comments (
                    comment_id TEXT PRIMARY KEY,
                    artifact_id TEXT NOT NULL,
                    parent_comment_id TEXT,
                    author TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    resolved INTEGER DEFAULT 0,
                    FOREIGN KEY (artifact_id) REFERENCES artifacts(artifact_id)
                )
            """)
            conn.commit()
        finally:
            conn.close()

        # Migration: add owner_id column if missing
        self._migrate_add_owner_id()

    def create_artifact(
        self,
        artifact_type: str,
        title: str,
        content: str,
        author: str = "system",
        owner_id: str = "",
        tags: Optional[List[str]] = None,
        message: str = "Initial version",
    ) -> ResearchArtifact:
        now = datetime.now(timezone.utc).isoformat()
        artifact_id = str(uuid.uuid4())[:8]
        version_id = str(uuid.uuid4())[:8]
        tags = tags or []

        artifact = ResearchArtifact(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            title=title,
            content=content,
            current_version=1,
            created_at=now,
            updated_at=now,
            author=author,
            owner_id=owner_id,
            tags=tags,
        )

        version = ArtifactVersion(
            version_id=version_id,
            artifact_id=artifact_id,
            content=content,
            message=message,
            created_at=now,
            author=author,
            tags=tags,
        )
        artifact.versions = [version]

        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                "INSERT INTO artifacts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (artifact_id, artifact_type, title, content, 1, now, now, author, json.dumps(tags), owner_id),
            )
            conn.execute(
                "INSERT INTO versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (version_id, artifact_id, 1, content, message, now, author, "", json.dumps(tags)),
            )
            conn.commit()
        finally:
            conn.close()

        logger.info(f"Created artifact {artifact_id} (v1): {title}")
        return artifact

    def update_artifact(
        self,
        artifact_id: str,
        content: str,
        message: str = "Updated",
        author: str = "system",
        tags: Optional[List[str]] = None,
    ) -> Optional[ArtifactVersion]:
        artifact = self.get_artifact(artifact_id)
        if not artifact:
            return None

        now = datetime.now(timezone.utc).isoformat()
        new_version = artifact.current_version + 1
        diff = self._compute_diff(artifact.content, content)
        version_id = str(uuid.uuid4())[:8]
        tags = tags or artifact.tags

        version = ArtifactVersion(
            version_id=version_id,
            artifact_id=artifact_id,
            content=content,
            message=message,
            created_at=now,
            author=author,
            diff_from_previous=diff,
            tags=tags,
        )

        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                "UPDATE artifacts SET content=?, current_version=?, updated_at=?, tags=? WHERE artifact_id=?",
                (content, new_version, now, json.dumps(tags), artifact_id),
            )
            conn.execute(
                "INSERT INTO versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (version_id, artifact_id, new_version, content, message, now, author, diff, json.dumps(tags)),
            )
            conn.commit()
        finally:
            conn.close()

        logger.info(f"Updated artifact {artifact_id} to v{new_version}")
        return version

    def get_artifact(self, artifact_id: str) -> Optional[ResearchArtifact]:
        conn = sqlite3.connect(self._db_path)
        try:
            row = conn.execute(
                "SELECT * FROM artifacts WHERE artifact_id=?", (artifact_id,)
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return None
        return ResearchArtifact(
            artifact_id=row[0],
            artifact_type=row[1],
            title=row[2],
            content=row[3],
            current_version=row[4],
            created_at=row[5],
            updated_at=row[6],
            author=row[7],
            tags=json.loads(row[8]) if row[8] else [],
            owner_id=row[9] if len(row) > 9 else "",
        )

    def get_history(self, artifact_id: str) -> List[ArtifactVersion]:
        conn = sqlite3.connect(self._db_path)
        try:
            rows = conn.execute(
                "SELECT * FROM versions WHERE artifact_id=? ORDER BY version_number ASC",
                (artifact_id,),
            ).fetchall()
        finally:
            conn.close()
        return [
            ArtifactVersion(
                version_id=r[0],
                artifact_id=r[1],
                content=r[3],
                message=r[4],
                created_at=r[5],
                author=r[6],
                diff_from_previous=r[7],
                tags=json.loads(r[8]) if r[8] else [],
            )
            for r in rows
        ]

    def list_artifacts(
        self, artifact_type: Optional[str] = None, tag: Optional[str] = None, owner_id: Optional[str] = None
    ) -> List[ResearchArtifact]:
        conn = sqlite3.connect(self._db_path)
        try:
            if owner_id:
                rows = conn.execute(
                    "SELECT * FROM artifacts WHERE owner_id=? ORDER BY updated_at DESC",
                    (owner_id,),
                ).fetchall()
            elif artifact_type:
                rows = conn.execute(
                    "SELECT * FROM artifacts WHERE artifact_type=? ORDER BY updated_at DESC",
                    (artifact_type,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM artifacts ORDER BY updated_at DESC"
                ).fetchall()
        finally:
            conn.close()

        artifacts = []
        for r in rows:
            tags = json.loads(r[8]) if r[8] else []
            if tag and tag not in tags:
                continue
            artifacts.append(
                ResearchArtifact(
                    artifact_id=r[0],
                    artifact_type=r[1],
                    title=r[2],
                    content=r[3],
                    current_version=r[4],
                    created_at=r[5],
                    updated_at=r[6],
                    author=r[7],
                    tags=tags,
                    owner_id=r[9] if len(r) > 9 else "",
                )
            )
        return artifacts

    def add_comment(
        self,
        artifact_id: str,
        author: str,
        content: str,
        parent_comment_id: Optional[str] = None,
    ) -> dict:
        now = datetime.now(timezone.utc).isoformat()
        comment_id = str(uuid.uuid4())[:8]
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                "INSERT INTO comments VALUES (?, ?, ?, ?, ?, ?, ?)",
                (comment_id, artifact_id, parent_comment_id, author, content, now, 0),
            )
            conn.commit()
        finally:
            conn.close()
        return {
            "comment_id": comment_id,
            "artifact_id": artifact_id,
            "author": author,
            "content": content,
            "created_at": now,
        }

    def get_comments(self, artifact_id: str) -> List[dict]:
        conn = sqlite3.connect(self._db_path)
        try:
            rows = conn.execute(
                "SELECT * FROM comments WHERE artifact_id=? ORDER BY created_at ASC",
                (artifact_id,),
            ).fetchall()
        finally:
            conn.close()
        return [
            {
                "comment_id": r[0],
                "artifact_id": r[1],
                "parent_comment_id": r[2],
                "author": r[3],
                "content": r[4],
                "created_at": r[5],
                "resolved": bool(r[6]),
            }
            for r in rows
        ]

    def resolve_comment(self, comment_id: str) -> bool:
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute("UPDATE comments SET resolved=1 WHERE comment_id=?", (comment_id,))
            conn.commit()
            changes = conn.total_changes
        finally:
            conn.close()
        return changes > 0

    @staticmethod
    def _compute_diff(old: str, new: str) -> str:
        old_lines = old.splitlines(keepends=True)
        new_lines = new.splitlines(keepends=True)
        diff = difflib.unified_diff(old_lines, new_lines, lineterm="")
        return "".join(diff)

    def _migrate_add_owner_id(self) -> None:
        """Add owner_id column to existing artifacts table if missing."""
        conn = sqlite3.connect(self._db_path)
        try:
            cursor = conn.execute("PRAGMA table_info(artifacts)")
            columns = [row[1] for row in cursor.fetchall()]
            if "owner_id" not in columns:
                conn.execute("ALTER TABLE artifacts ADD COLUMN owner_id TEXT DEFAULT ''")
                conn.commit()
        finally:
            conn.close()
