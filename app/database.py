import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.models import Citation, ConversationDetail, ConversationMessage, ConversationSummary, IngestionJob


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


class ConversationStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    owner_id TEXT NOT NULL DEFAULT 'demo-user'
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    citations TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                    ON messages(conversation_id, id);
                CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    files_received INTEGER NOT NULL,
                    document_units INTEGER NOT NULL,
                    chunks_added INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(conversations)")}
            if "owner_id" not in columns:
                connection.execute("ALTER TABLE conversations ADD COLUMN owner_id TEXT NOT NULL DEFAULT 'demo-user'")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def save_exchange(
        self,
        conversation_id: str | None,
        question: str,
        answer: str,
        citations: list[Citation],
        owner_id: str = "demo-user",
    ) -> str:
        timestamp = now_iso()
        with self._connect() as connection:
            if conversation_id:
                exists = connection.execute(
                    "SELECT id FROM conversations WHERE id = ? AND owner_id = ?", (conversation_id, owner_id)
                ).fetchone()
                if not exists:
                    conversation_id = None
            if not conversation_id:
                conversation_id = uuid.uuid4().hex
                title = " ".join(question.split())[:72] or "New conversation"
                connection.execute(
                    "INSERT INTO conversations(id, title, created_at, updated_at, owner_id) VALUES (?, ?, ?, ?, ?)",
                    (conversation_id, title, timestamp, timestamp, owner_id),
                )
            connection.execute(
                "INSERT INTO messages(conversation_id, role, content, citations, created_at) VALUES (?, 'user', ?, '[]', ?)",
                (conversation_id, question, timestamp),
            )
            connection.execute(
                "INSERT INTO messages(conversation_id, role, content, citations, created_at) VALUES (?, 'assistant', ?, ?, ?)",
                (
                    conversation_id,
                    answer,
                    json.dumps([citation.model_dump() for citation in citations]),
                    timestamp,
                ),
            )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (timestamp, conversation_id),
            )
        return conversation_id

    def list_conversations(self, owner_id: str = "demo-user", limit: int = 50) -> list[ConversationSummary]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT c.*, COUNT(m.id) AS message_count
                FROM conversations c LEFT JOIN messages m ON m.conversation_id = c.id
                WHERE c.owner_id = ? GROUP BY c.id ORDER BY c.updated_at DESC LIMIT ?
                """,
                (owner_id, limit),
            ).fetchall()
        return [
            ConversationSummary(
                conversation_id=row["id"],
                title=row["title"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                message_count=row["message_count"],
            )
            for row in rows
        ]

    def get_conversation(self, conversation_id: str, owner_id: str = "demo-user") -> ConversationDetail | None:
        with self._connect() as connection:
            conversation = connection.execute(
                "SELECT c.*, COUNT(m.id) AS message_count FROM conversations c LEFT JOIN messages m ON m.conversation_id = c.id WHERE c.id = ? AND c.owner_id = ? GROUP BY c.id",
                (conversation_id, owner_id),
            ).fetchone()
            if not conversation:
                return None
            rows = connection.execute(
                "SELECT role, content, citations, created_at FROM messages WHERE conversation_id = ? ORDER BY id",
                (conversation_id,),
            ).fetchall()
        messages: list[ConversationMessage] = []
        for row in rows:
            payload: dict[str, Any] = dict(row)
            payload["citations"] = [Citation(**item) for item in json.loads(payload["citations"])]
            messages.append(ConversationMessage(**payload))
        return ConversationDetail(
            conversation_id=conversation["id"],
            title=conversation["title"],
            created_at=conversation["created_at"],
            updated_at=conversation["updated_at"],
            message_count=conversation["message_count"],
            messages=messages,
        )

    def rename_conversation(self, conversation_id: str, title: str, owner_id: str = "demo-user") -> bool:
        with self._connect() as connection:
            result = connection.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ? AND owner_id = ?",
                (title.strip(), now_iso(), conversation_id, owner_id),
            )
        return result.rowcount == 1

    def delete_conversation(self, conversation_id: str, owner_id: str = "demo-user") -> bool:
        with self._connect() as connection:
            result = connection.execute(
                "DELETE FROM conversations WHERE id = ? AND owner_id = ?",
                (conversation_id, owner_id),
            )
        return result.rowcount == 1

    def create_job(self, job_id: str, files_received: int, document_units: int) -> IngestionJob:
        timestamp = now_iso()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO ingestion_jobs(id, status, files_received, document_units, created_at, updated_at) VALUES (?, 'queued', ?, ?, ?, ?)",
                (job_id, files_received, document_units, timestamp, timestamp),
            )
        return self.get_job(job_id)  # type: ignore[return-value]

    def update_job(self, job_id: str, status: str, chunks_added: int = 0, error: str | None = None) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE ingestion_jobs SET status = ?, chunks_added = ?, error = ?, updated_at = ? WHERE id = ?",
                (status, chunks_added, error, now_iso(), job_id),
            )

    def get_job(self, job_id: str) -> IngestionJob | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM ingestion_jobs WHERE id = ?", (job_id,)).fetchone()
        if not row:
            return None
        return IngestionJob(
            job_id=row["id"],
            status=row["status"],
            files_received=row["files_received"],
            document_units=row["document_units"],
            chunks_added=row["chunks_added"],
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
