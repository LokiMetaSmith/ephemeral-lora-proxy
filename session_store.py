import sqlite3
import threading
import time
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class SessionState:
    session_id: str
    active_lora_scale: float = 1.0
    generation: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class SessionStore:
    def __init__(self, db_path: str = "/home/loki/ephemeral-lora-proxy/sessions.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._memory_sessions: Dict[str, SessionState] = {}
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._lock:
            with self._get_conn() as conn:
                conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    active_lora_scale REAL DEFAULT 1.0,
                    generation INTEGER DEFAULT 0,
                    created_at REAL,
                    updated_at REAL
                )
                """)
                conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    extra_json TEXT,
                    is_consolidated INTEGER DEFAULT 0,
                    created_at REAL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
                """)
                try:
                    conn.execute("ALTER TABLE messages ADD COLUMN extra_json TEXT")
                except sqlite3.OperationalError:
                    pass
                conn.commit()

    @staticmethod
    def _row_to_msg(r: sqlite3.Row) -> Dict[str, Any]:
        msg = {
            "id": r["id"],
            "role": r["role"],
            "content": r["content"],
            "is_consolidated": r["is_consolidated"],
        }
        if "extra_json" in r.keys() and r["extra_json"]:
            try:
                extra = json.loads(r["extra_json"])
                if isinstance(extra, dict):
                    msg.update(extra)
            except Exception:
                pass
        return msg

    def get_or_create_session(self, session_id: str) -> SessionState:
        with self._lock:
            if session_id in self._memory_sessions:
                return self._memory_sessions[session_id]

            with self._get_conn() as conn:
                cur = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
                row = cur.fetchone()
                now = time.time()
                if row:
                    state = SessionState(
                        session_id=row["session_id"],
                        active_lora_scale=row["active_lora_scale"],
                        generation=row["generation"],
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                    )
                else:
                    conn.execute(
                        "INSERT INTO sessions (session_id, active_lora_scale, generation, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                        (session_id, 1.0, 0, now, now)
                    )
                    conn.commit()
                    state = SessionState(session_id=session_id, active_lora_scale=1.0, generation=0, created_at=now, updated_at=now)

                self._memory_sessions[session_id] = state
                return state

    def add_message(
        self,
        session_id: str,
        role: str,
        content: Optional[Any] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> int:
        self.get_or_create_session(session_id)
        now = time.time()
        extra_str = json.dumps(extra) if extra else None
        # In sqlite text content column, store string or None
        text_content = content if isinstance(content, str) else (None if content is None else json.dumps(content))
        with self._lock:
            with self._get_conn() as conn:
                cur = conn.execute(
                    "INSERT INTO messages (session_id, role, content, extra_json, is_consolidated, created_at) VALUES (?, ?, ?, ?, 0, ?)",
                    (session_id, role, text_content, extra_str, now)
                )
                msg_id = cur.lastrowid
                conn.execute("UPDATE sessions SET updated_at = ? WHERE session_id = ?", (now, session_id))
                conn.commit()
                if session_id in self._memory_sessions:
                    self._memory_sessions[session_id].updated_at = now
                return msg_id

    def get_active_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Returns unconsolidated messages (is_consolidated != 1, including pending consolidation)
        plus any initial system message. These are retained in the active prompt payload until
        consolidation has fully completed and hot-reloaded into the adapter.
        """
        with self._lock:
            with self._get_conn() as conn:
                cur = conn.execute(
                    "SELECT id, role, content, extra_json, is_consolidated FROM messages WHERE session_id = ? ORDER BY id ASC",
                    (session_id,)
                )
                rows = cur.fetchall()
                if not rows:
                    return []

                result = []
                # Keep system prompt if first message
                first = rows[0]
                if first["role"] == "system":
                    result.append(self._row_to_msg(first))
                    rows = rows[1:]

                for r in rows:
                    if r["is_consolidated"] != 1:
                        result.append(self._row_to_msg(r))
                return result

    def get_unconsolidated_candidates(self, session_id: str) -> List[Dict[str, Any]]:
        """Returns messages eligible for eviction (is_consolidated == 0, excluding system prompt)."""
        with self._lock:
            with self._get_conn() as conn:
                cur = conn.execute(
                    "SELECT id, role, content, extra_json, is_consolidated FROM messages WHERE session_id = ? AND is_consolidated = 0 AND role != 'system' ORDER BY id ASC",
                    (session_id,)
                )
                return [self._row_to_msg(r) for r in cur.fetchall()]

    def mark_messages_pending(self, session_id: str, msg_ids: List[int]):
        """Marks messages as pending consolidation (is_consolidated = 2) to prevent re-enqueue."""
        if not msg_ids:
            return
        with self._lock:
            with self._get_conn() as conn:
                placeholders = ",".join("?" * len(msg_ids))
                conn.execute(
                    f"UPDATE messages SET is_consolidated = 2 WHERE session_id = ? AND id IN ({placeholders})",
                    [session_id] + msg_ids
                )
                conn.commit()

    def reset_pending_messages(self, session_id: str, msg_ids: List[int]):
        """Resets pending messages back to unconsolidated (is_consolidated = 0) upon worker failure."""
        if not msg_ids:
            return
        with self._lock:
            with self._get_conn() as conn:
                placeholders = ",".join("?" * len(msg_ids))
                conn.execute(
                    f"UPDATE messages SET is_consolidated = 0 WHERE session_id = ? AND id IN ({placeholders}) AND is_consolidated = 2",
                    [session_id] + msg_ids
                )
                conn.commit()

    def get_all_messages(self, session_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            with self._get_conn() as conn:
                cur = conn.execute(
                    "SELECT id, role, content, extra_json, is_consolidated FROM messages WHERE session_id = ? ORDER BY id ASC",
                    (session_id,)
                )
                return [self._row_to_msg(r) for r in cur.fetchall()]

    def mark_messages_consolidated(self, session_id: str, msg_ids: List[int]):
        if not msg_ids:
            return
        with self._lock:
            with self._get_conn() as conn:
                placeholders = ",".join("?" * len(msg_ids))
                conn.execute(
                    f"UPDATE messages SET is_consolidated = 1 WHERE session_id = ? AND id IN ({placeholders})",
                    [session_id] + msg_ids
                )
                conn.commit()

    def increment_generation(self, session_id: str) -> int:
        now = time.time()
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    "UPDATE sessions SET generation = generation + 1, updated_at = ? WHERE session_id = ?",
                    (now, session_id)
                )
                cur = conn.execute("SELECT generation FROM sessions WHERE session_id = ?", (session_id,))
                row = cur.fetchone()
                gen = row["generation"] if row else 1
                conn.commit()
                if session_id in self._memory_sessions:
                    self._memory_sessions[session_id].generation = gen
                    self._memory_sessions[session_id].updated_at = now
                return gen

    def set_scale(self, session_id: str, scale: float):
        now = time.time()
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    "UPDATE sessions SET active_lora_scale = ?, updated_at = ? WHERE session_id = ?",
                    (scale, now, session_id)
                )
                conn.commit()
                if session_id in self._memory_sessions:
                    self._memory_sessions[session_id].active_lora_scale = scale
                    self._memory_sessions[session_id].updated_at = now

    def reset_session(self, session_id: str):
        now = time.time()
        with self._lock:
            with self._get_conn() as conn:
                conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
                conn.execute(
                    "UPDATE sessions SET active_lora_scale = 1.0, generation = 0, updated_at = ? WHERE session_id = ?",
                    (now, session_id)
                )
                conn.commit()
                if session_id in self._memory_sessions:
                    self._memory_sessions[session_id].active_lora_scale = 1.0
                    self._memory_sessions[session_id].generation = 0
                    self._memory_sessions[session_id].updated_at = now
