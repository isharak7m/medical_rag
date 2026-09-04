"""
Collaboration service — shared workspaces and real-time comments.
Provides workspace management, member access, and comment threads.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field

from utils.logger import get_logger

logger = get_logger(__name__)


class WorkspaceMember(BaseModel):
    member_id: str
    workspace_id: str
    username: str
    role: str = "viewer"  # viewer, editor, admin
    joined_at: str


class Workspace(BaseModel):
    workspace_id: str
    name: str
    description: str
    created_by: str
    created_at: str
    updated_at: str
    members: List[WorkspaceMember] = Field(default_factory=list)
    shared_artifacts: List[str] = Field(default_factory=list)  # artifact IDs


class CollabComment(BaseModel):
    comment_id: str
    workspace_id: str
    artifact_id: Optional[str] = None
    parent_comment_id: Optional[str] = None
    author: str
    content: str
    created_at: str
    resolved: bool = False
    mentions: List[str] = Field(default_factory=list)


class CollaborationStore:
    """
    SQLite-backed collaboration store for workspaces, members, and comments.
    """

    def __init__(self, db_path: str = "collaboration.db") -> None:
        self._db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS workspaces (
                    workspace_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    shared_artifacts TEXT DEFAULT '[]'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS workspace_members (
                    member_id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    username TEXT NOT NULL,
                    role TEXT DEFAULT 'viewer',
                    joined_at TEXT NOT NULL,
                    FOREIGN KEY (workspace_id) REFERENCES workspaces(workspace_id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS collab_comments (
                    comment_id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    artifact_id TEXT,
                    parent_comment_id TEXT,
                    author TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    resolved INTEGER DEFAULT 0,
                    mentions TEXT DEFAULT '[]',
                    FOREIGN KEY (workspace_id) REFERENCES workspaces(workspace_id)
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def create_workspace(
        self, name: str, description: str, created_by: str
    ) -> Workspace:
        now = datetime.now(timezone.utc).isoformat()
        ws_id = str(uuid.uuid4())[:8]

        ws = Workspace(
            workspace_id=ws_id,
            name=name,
            description=description,
            created_by=created_by,
            created_at=now,
            updated_at=now,
        )

        # Auto-add creator as admin
        member = self.add_member(ws_id, created_by, "admin")
        ws.members = [member]

        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                "INSERT INTO workspaces VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ws_id, name, description, created_by, now, now, "[]"),
            )
            conn.commit()
        finally:
            conn.close()

        logger.info(f"Created workspace '{name}' ({ws_id}) by {created_by}")
        return ws

    def get_workspace(self, workspace_id: str) -> Optional[Workspace]:
        conn = sqlite3.connect(self._db_path)
        try:
            row = conn.execute(
                "SELECT * FROM workspaces WHERE workspace_id=?", (workspace_id,)
            ).fetchone()
            if not row:
                return None
            members = self._get_members(workspace_id, conn)
        finally:
            conn.close()

        return Workspace(
            workspace_id=row[0],
            name=row[1],
            description=row[2],
            created_by=row[3],
            created_at=row[4],
            updated_at=row[5],
            shared_artifacts=json.loads(row[6]) if row[6] else [],
            members=members,
        )

    def list_workspaces(self, username: Optional[str] = None) -> List[Workspace]:
        conn = sqlite3.connect(self._db_path)
        try:
            if username:
                rows = conn.execute(
                    """SELECT w.* FROM workspaces w
                       JOIN workspace_members m ON w.workspace_id = m.workspace_id
                       WHERE m.username = ?
                       ORDER BY w.updated_at DESC""",
                    (username,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM workspaces ORDER BY updated_at DESC"
                ).fetchall()

            results = []
            for r in rows:
                members = self._get_members(r[0], conn)
                results.append(
                    Workspace(
                        workspace_id=r[0],
                        name=r[1],
                        description=r[2],
                        created_by=r[3],
                        created_at=r[4],
                        updated_at=r[5],
                        shared_artifacts=json.loads(r[6]) if r[6] else [],
                        members=members,
                    )
                )
        finally:
            conn.close()
        return results

    def add_member(
        self, workspace_id: str, username: str, role: str = "viewer"
    ) -> WorkspaceMember:
        now = datetime.now(timezone.utc).isoformat()
        member_id = str(uuid.uuid4())[:8]

        member = WorkspaceMember(
            member_id=member_id,
            workspace_id=workspace_id,
            username=username,
            role=role,
            joined_at=now,
        )

        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                "INSERT OR IGNORE INTO workspace_members VALUES (?, ?, ?, ?, ?)",
                (member_id, workspace_id, username, role, now),
            )
            conn.commit()
        finally:
            conn.close()
        return member

    def share_artifact(self, workspace_id: str, artifact_id: str) -> bool:
        conn = sqlite3.connect(self._db_path)
        try:
            row = conn.execute(
                "SELECT shared_artifacts FROM workspaces WHERE workspace_id=?",
                (workspace_id,),
            ).fetchone()
            if not row:
                return False

            artifacts = json.loads(row[0]) if row[0] else []
            if artifact_id not in artifacts:
                artifacts.append(artifact_id)
                now = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    "UPDATE workspaces SET shared_artifacts=?, updated_at=? WHERE workspace_id=?",
                    (json.dumps(artifacts), now, workspace_id),
                )
                conn.commit()
        finally:
            conn.close()
        return True

    def add_comment(
        self,
        workspace_id: str,
        author: str,
        content: str,
        artifact_id: Optional[str] = None,
        parent_comment_id: Optional[str] = None,
    ) -> CollabComment:
        now = datetime.now(timezone.utc).isoformat()
        comment_id = str(uuid.uuid4())[:8]

        # Extract @mentions
        import re
        mentions = re.findall(r"@(\w+)", content)

        comment = CollabComment(
            comment_id=comment_id,
            workspace_id=workspace_id,
            artifact_id=artifact_id,
            parent_comment_id=parent_comment_id,
            author=author,
            content=content,
            created_at=now,
            mentions=mentions,
        )

        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                "INSERT INTO collab_comments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (comment_id, workspace_id, artifact_id, parent_comment_id, author, content, now, 0, json.dumps(mentions)),
            )
            conn.commit()
        finally:
            conn.close()

        logger.info(f"Comment {comment_id} in workspace {workspace_id} by {author}")
        return comment

    def get_comments(
        self,
        workspace_id: str,
        artifact_id: Optional[str] = None,
    ) -> List[CollabComment]:
        conn = sqlite3.connect(self._db_path)
        try:
            if artifact_id:
                rows = conn.execute(
                    "SELECT * FROM collab_comments WHERE workspace_id=? AND artifact_id=? ORDER BY created_at ASC",
                    (workspace_id, artifact_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM collab_comments WHERE workspace_id=? ORDER BY created_at ASC",
                    (workspace_id,),
                ).fetchall()
        finally:
            conn.close()
        return [
            CollabComment(
                comment_id=r[0],
                workspace_id=r[1],
                artifact_id=r[2],
                parent_comment_id=r[3],
                author=r[4],
                content=r[5],
                created_at=r[6],
                resolved=bool(r[7]),
                mentions=json.loads(r[8]) if r[8] else [],
            )
            for r in rows
        ]

    def resolve_comment(self, comment_id: str) -> bool:
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                "UPDATE collab_comments SET resolved=1 WHERE comment_id=?",
                (comment_id,),
            )
            conn.commit()
            changed = conn.total_changes
        finally:
            conn.close()
        return changed > 0

    def _get_members(self, workspace_id: str, conn: sqlite3.Connection) -> List[WorkspaceMember]:
        rows = conn.execute(
            "SELECT * FROM workspace_members WHERE workspace_id=?",
            (workspace_id,),
        ).fetchall()
        return [
            WorkspaceMember(
                member_id=r[0],
                workspace_id=r[1],
                username=r[2],
                role=r[3],
                joined_at=r[4],
            )
            for r in rows
        ]
