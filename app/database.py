import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.models import Citation, ConversationDetail, ConversationMessage, ConversationSummary


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
                    updated_at TEXT NOT NULL
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
                """
            )

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
    ) -> str:
        timestamp = now_iso()
        with self._connect() as connection:
            if conversation_id:
                exists = connection.execute(
                    "SELECT id FROM conversations WHERE id = ?", (conversation_id,)
                ).fetchone()
                if not exists:
                    conversation_id = None
            if not conversation_id:
                conversation_id = uuid.uuid4().hex
                title = " ".join(question.split())[:72] or "New conversation"
                connection.execute(
                    "INSERT INTO conversations VALUES (?, ?, ?, ?)",
                    (conversation_id, title, timestamp, timestamp),
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

    def list_conversations(self, limit: int = 50) -> list[ConversationSummary]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT c.*, COUNT(m.id) AS message_count
                FROM conversations c LEFT JOIN messages m ON m.conversation_id = c.id
                GROUP BY c.id ORDER BY c.updated_at DESC LIMIT ?
                """,
                (limit,),
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

    def get_conversation(self, conversation_id: str) -> ConversationDetail | None:
        with self._connect() as connection:
            conversation = connection.execute(
                "SELECT c.*, COUNT(m.id) AS message_count FROM conversations c LEFT JOIN messages m ON m.conversation_id = c.id WHERE c.id = ? GROUP BY c.id",
                (conversation_id,),
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
