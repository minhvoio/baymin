"""
Persistence layer for conversation state.

This module provides optional persistence support for saving and
loading conversation states. Currently supports SQLite for simple
local storage, but the design allows easy extension to other backends.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path
import json
import sqlite3


class SQLitePersistence:
    """
    SQLite-based persistence for conversation state.

    Stores conversation states in a local SQLite database, enabling
    conversation resumption across sessions.

    Example:
        ```python
        from bn_helpers.langgraph import ConversationManager
        from bn_helpers.langgraph.persistence import SQLitePersistence

        # Setup persistence
        db = SQLitePersistence("conversations.db")

        # Create and use conversation
        conv = ConversationManager(net)
        answer = conv.ask("Is A connected to B?")

        # Save for later
        db.save(conv.conversation_id, conv.save_state(), "ChestClinic")

        # Later, restore
        saved = db.load(conv.conversation_id)
        conv.load_state(saved)
        ```
    """

    def __init__(self, db_path: str = "bn_conversations.db"):
        """
        Initialize SQLite persistence.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
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
        """
        Save conversation state to database.

        Args:
            conversation_id: Unique conversation identifier
            state: State dict from ConversationManager.save_state()
            network_name: Name of the BN (for filtering)
        """
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
        """
        Load conversation state from database.

        Args:
            conversation_id: Unique conversation identifier

        Returns:
            State dict if found, None otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT state_json FROM conversations WHERE id = ?",
                (conversation_id,)
            ).fetchone()

            if row:
                return json.loads(row[0])

        return None

    def delete(self, conversation_id: str) -> bool:
        """
        Delete a conversation from database.

        Args:
            conversation_id: Unique conversation identifier

        Returns:
            True if deleted, False if not found
        """
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
        """
        List saved conversations.

        Args:
            network_name: Filter by network name (optional)
            limit: Maximum number of results

        Returns:
            List of conversation metadata dicts
        """
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
        """
        Get a summary of a conversation without loading full state.

        Args:
            conversation_id: Unique conversation identifier

        Returns:
            Summary dict or None if not found
        """
        state = self.load(conversation_id)
        if not state:
            return None

        metadata = state.get("metadata", {})
        messages = state.get("messages", [])

        # Extract user questions
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
    """
    In-memory persistence for testing and ephemeral usage.

    Stores conversations in a dict - useful for testing or when
    persistence isn't needed beyond the current session.
    """

    def __init__(self):
        """Initialize empty storage."""
        self._storage: Dict[str, Dict[str, Any]] = {}
        self._metadata: Dict[str, Dict[str, str]] = {}

    def save(
        self,
        conversation_id: str,
        state: Dict[str, Any],
        network_name: str
    ) -> None:
        """Save state to memory."""
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
        """Load state from memory."""
        return self._storage.get(conversation_id)

    def delete(self, conversation_id: str) -> bool:
        """Delete from memory."""
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
        """List stored conversations."""
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

        # Sort by updated_at descending
        results.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return results[:limit]

    def clear(self) -> None:
        """Clear all stored conversations."""
        self._storage.clear()
        self._metadata.clear()
