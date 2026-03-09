from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path
import json, sqlite3


class SQLitePersistence:
    """SQLite-based persistence for conversation state."""

    def __init__(self, db_path: str = "bn_conversations.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    network_name TEXT,
                    state_json TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_network_name
                ON conversations(network_name)
            """)

    def save(
        self,
        conversation_id: str,
        state: Dict[str, Any],
        network_name: str
    ) -> None:
        created_at = state.get("metadata", {}).get(
            "created_at",
            datetime.now().isoformat()
        )
        updated_at = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO conversations
                (id, network_name, state_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                conversation_id,
                network_name,
                json.dumps(state, ensure_ascii=False),
                created_at,
                updated_at,
            ))

    def load(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT state_json FROM conversations WHERE id = ?",
                (conversation_id,)
            ).fetchone()

            if row:
                return json.loads(row[0])

        return None

    def delete(self, conversation_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM conversations WHERE id = ?",
                (conversation_id,)
            )
            return cursor.rowcount > 0

    def list_conversations(
        self,
        network_name: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            if network_name:
                rows = conn.execute("""
                    SELECT id, network_name, created_at, updated_at
                    FROM conversations
                    WHERE network_name = ?
                    ORDER BY updated_at DESC
                    LIMIT ?
                """, (network_name, limit)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT id, network_name, created_at, updated_at
                    FROM conversations
                    ORDER BY updated_at DESC
                    LIMIT ?
                """, (limit,)).fetchall()

            return [
                {
                    "id": r[0],
                    "network_name": r[1],
                    "created_at": r[2],
                    "updated_at": r[3],
                }
                for r in rows
            ]

    def get_conversation_summary(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        state = self.load(conversation_id)
        if not state:
            return None

        metadata = state.get("metadata", {})
        messages = state.get("messages", [])

        questions = [
            m.get("content", "")[:100]
            for m in messages
            if m.get("type") == "HumanMessage"
        ]

        return {
            "id": conversation_id,
            "network_name": state.get("network_info", {}).get("name", "Unknown"),
            "turn_count": metadata.get("turn_count", 0),
            "created_at": metadata.get("created_at"),
            "updated_at": metadata.get("updated_at"),
            "questions": questions,
            "cache_size": len(state.get("tool_results_cache", {})),
            "artifact_count": len(state.get("artifacts", {})),
        }


class InMemoryPersistence:
    """In-memory persistence for testing and ephemeral usage."""

    def __init__(self):
        self._storage: Dict[str, Dict[str, Any]] = {}
        self._metadata: Dict[str, Dict[str, str]] = {}

    def save(
        self,
        conversation_id: str,
        state: Dict[str, Any],
        network_name: str
    ) -> None:
        self._storage[conversation_id] = state
        self._metadata[conversation_id] = {
            "network_name": network_name,
            "created_at": state.get("metadata", {}).get(
                "created_at",
                datetime.now().isoformat()
            ),
            "updated_at": datetime.now().isoformat(),
        }

    def load(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        return self._storage.get(conversation_id)

    def delete(self, conversation_id: str) -> bool:
        if conversation_id in self._storage:
            del self._storage[conversation_id]
            del self._metadata[conversation_id]
            return True
        return False

    def list_conversations(
        self,
        network_name: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        results = []
        for conv_id, meta in self._metadata.items():
            if network_name and meta.get("network_name") != network_name:
                continue
            results.append({
                "id": conv_id,
                "network_name": meta.get("network_name"),
                "created_at": meta.get("created_at"),
                "updated_at": meta.get("updated_at"),
            })

        results.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return results[:limit]

    def clear(self) -> None:
        self._storage.clear()
        self._metadata.clear()
