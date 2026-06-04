import sqlite3
import json
from pathlib import Path
from typing import Any


class VolitionalDB:
    """Volitional 插件的 SQLite 数据库工具。

    数据文件位于插件数据目录下的 volitional.db。
    路径由外部通过 data_dir 参数传入。
    """

    def __init__(self, data_dir: Path):
        self._db_path = data_dir / "volitional.db"
        self._conn: sqlite3.Connection | None = None

    def _ensure_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self._db_path), check_same_thread=False
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _cursor(self):
        return self._ensure_conn().cursor()

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # ------ 建表 ------ #

    def init_tables(self):
        c = self._cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                conv_id TEXT PRIMARY KEY,
                umo TEXT NOT NULL,
                bot_name TEXT DEFAULT '',
                title TEXT,
                created_at TEXT DEFAULT (datetime('now', 'localtime')),
                updated_at TEXT DEFAULT (datetime('now', 'localtime'))
            )
        """)
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_conv_umo
            ON conversations(umo)
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conv_id TEXT NOT NULL,
                seq INTEGER NOT NULL DEFAULT 0,
                role TEXT NOT NULL,
                content TEXT NOT NULL DEFAULT '',
                tool_calls TEXT,
                tool_call_id TEXT,
                name TEXT,
                created_at TEXT DEFAULT (datetime('now', 'localtime')),
                FOREIGN KEY (conv_id) REFERENCES conversations(conv_id)
                    ON DELETE CASCADE
            )
        """)
        self._migrate_add_msg_chat_columns(c)
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_msg_conv_seq
            ON messages(conv_id, seq)
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS judgment_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                umo TEXT NOT NULL,
                chat_type TEXT DEFAULT '',
                chat_id TEXT DEFAULT '',
                conv_id TEXT,
                sender_name TEXT,
                message TEXT,
                overall REAL,
                speaker_target_clarity REAL,
                privacy_safety_risk REAL,
                relevance REAL,
                user_intent_clarity REAL,
                replyability REAL,
                context_completeness REAL,
                turn_idleness REAL,
                emotional_suitability REAL,
                intervention_naturalness REAL,
                group_atmosphere_fit REAL,
                should_reply INTEGER,
                reason TEXT,
                created_at TEXT DEFAULT (datetime('now', 'localtime'))
            )
        """)
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_judgment_umo
            ON judgment_log(umo)
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS kv_store (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT DEFAULT (datetime('now', 'localtime'))
            )
        """)
        self._ensure_conn().commit()

        self._migrate_add_chat_columns(c)
        self._migrate_v2_metrics(c)

    def _migrate_add_chat_columns(self, c):
        try:
            c.execute("ALTER TABLE judgment_log ADD COLUMN chat_type TEXT DEFAULT ''")
        except Exception:
            pass
        try:
            c.execute("ALTER TABLE judgment_log ADD COLUMN chat_id TEXT DEFAULT ''")
        except Exception:
            pass
        self._ensure_conn().commit()

    def _migrate_v2_metrics(self, c):
        cols = [
            "speaker_target_clarity", "privacy_safety_risk",
            "user_intent_clarity", "context_completeness",
            "turn_idleness", "intervention_naturalness",
            "group_atmosphere_fit",
        ]
        for col in cols:
            try:
                c.execute(f"ALTER TABLE judgment_log ADD COLUMN {col} REAL DEFAULT 0.0")
            except Exception:
                pass
        self._ensure_conn().commit()

    def _migrate_add_msg_chat_columns(self, c):
        try:
            c.execute("ALTER TABLE messages ADD COLUMN chat_type TEXT DEFAULT ''")
        except Exception:
            pass
        try:
            c.execute("ALTER TABLE messages ADD COLUMN chat_id TEXT DEFAULT ''")
        except Exception:
            pass
        try:
            c.execute("ALTER TABLE messages ADD COLUMN sender_name TEXT DEFAULT ''")
        except Exception:
            pass
        self._ensure_conn().commit()

    @staticmethod
    def parse_umo(umo: str):
        parts = umo.split(":", 2)
        if len(parts) < 3:
            return "", ""
        msg_type = parts[1]
        chat_id = parts[2]
        if msg_type == "GroupMessage":
            chat_type = "群聊"
        elif msg_type == "FriendMessage":
            chat_type = "私聊"
        else:
            chat_type = "其他"
        return chat_type, chat_id

    # ------ 对话管理 ------ #

    def create_conversation(self, conv_id: str, umo: str, title: str = ""):
        c = self._cursor()
        c.execute(
            "INSERT INTO conversations (conv_id, umo, title) VALUES (?, ?, ?)",
            (conv_id, umo, title),
        )
        self._ensure_conn().commit()

    # ------ 消息管理 (OpenAI 格式) ------ #

    def add_message(
        self,
        conv_id: str,
        role: str,
        content: str,
        seq: int | None = None,
        tool_calls: list[dict] | None = None,
        tool_call_id: str | None = None,
        name: str | None = None,
        chat_type: str = "",
        chat_id: str = "",
        sender_name: str = "",
    ):
        c = self._cursor()
        exists = c.execute(
            "SELECT 1 FROM conversations WHERE conv_id = ?", (conv_id,)
        ).fetchone()
        if not exists:
            c.execute(
                "INSERT INTO conversations (conv_id, umo) VALUES (?, ?)",
                (conv_id, conv_id),
            )
        if seq is None:
            max_seq = c.execute(
                "SELECT COALESCE(MAX(seq), -1) FROM messages WHERE conv_id = ?",
                (conv_id,),
            ).fetchone()[0]
            seq = max_seq + 1

        c.execute(
            """INSERT INTO messages (conv_id, seq, role, content, tool_calls, tool_call_id, name, chat_type, chat_id, sender_name)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (conv_id, seq, role, content,
             json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None,
             tool_call_id, name, chat_type, chat_id, sender_name),
        )
        self._ensure_conn().execute(
            "UPDATE conversations SET updated_at = datetime('now', 'localtime') "
            "WHERE conv_id = ?",
            (conv_id,),
        )
        self._ensure_conn().commit()

    def get_messages(self, conv_id: str, limit: int | None = None) -> list[dict]:
        c = self._cursor()
        if limit:
            rows = c.execute(
                """SELECT role, content, tool_calls, tool_call_id, name, created_at, sender_name
                   FROM messages WHERE conv_id = ?
                   ORDER BY seq DESC LIMIT ?""",
                (conv_id, limit),
            ).fetchall()
            rows = list(reversed(rows))
        else:
            rows = c.execute(
                """SELECT role, content, tool_calls, tool_call_id, name, created_at, sender_name
                   FROM messages WHERE conv_id = ?
                   ORDER BY seq""",
                (conv_id,),
            ).fetchall()
        result = []
        for r in rows:
            msg = {"role": r[0], "content": r[1], "time": r[5], "sender_name": r[6] or ""}
            if r[2]:
                msg["tool_calls"] = json.loads(r[2])
            if r[3]:
                msg["tool_call_id"] = r[3]
            if r[4]:
                msg["name"] = r[4]
            result.append(msg)
        return result

    def get_conversation_list(self) -> list[dict]:
        c = self._cursor()
        rows = c.execute(
            """SELECT c.conv_id, c.umo, c.created_at, c.updated_at,
                      COALESCE(m.chat_type, '') as chat_type,
                      COALESCE(m.chat_id, '') as chat_id,
                      COALESCE(cnt.cnt, 0) as msg_count
               FROM conversations c
               LEFT JOIN (
                   SELECT DISTINCT conv_id, chat_type, chat_id FROM messages
                   WHERE chat_type != ''
               ) m ON m.conv_id = c.conv_id
               LEFT JOIN (
                   SELECT conv_id, COUNT(*) as cnt FROM messages GROUP BY conv_id
               ) cnt ON cnt.conv_id = c.conv_id
               ORDER BY c.updated_at DESC"""
        ).fetchall()
        return [
            {
                "conv_id": r[0], "umo": r[1],
                "created_at": r[2], "updated_at": r[3],
                "chat_type": r[4] or "其他", "chat_id": r[5] or r[0],
                "msg_count": r[6],
            }
            for r in rows
        ]

    def get_messages_detail(self, conv_id: str, limit: int = 500) -> dict:
        c = self._cursor()
        conv = c.execute(
            "SELECT conv_id, umo, created_at FROM conversations WHERE conv_id = ?",
            (conv_id,),
        ).fetchone()
        if not conv:
            return {"conv_id": conv_id, "chat_type": "未知", "chat_id": conv_id, "messages": []}

        meta = {"chat_type": "其他", "chat_id": conv[0]}
        meta_row = c.execute(
            "SELECT chat_type, chat_id FROM messages WHERE conv_id = ? AND chat_type != '' LIMIT 1",
            (conv_id,),
        ).fetchone()
        if meta_row:
            meta["chat_type"] = meta_row[0]
            meta["chat_id"] = meta_row[1]

        total = self.count_messages(conv_id)
        msgs = self.get_messages(conv_id, limit=limit)
        return {
            "conv_id": conv[0],
            "umo": conv[1],
            "created_at": conv[2],
            "chat_type": meta["chat_type"],
            "chat_id": meta["chat_id"],
            "total": total,
            "messages": msgs,
        }

    def get_messages_as_openai_json(self, conv_id: str) -> str:
        return json.dumps(self.get_messages(conv_id), ensure_ascii=False)

    def count_messages(self, conv_id: str) -> int:
        c = self._cursor()
        return c.execute(
            "SELECT COUNT(*) FROM messages WHERE conv_id = ?", (conv_id,)
        ).fetchone()[0]

    def delete_conversation(self, conv_id: str):
        c = self._cursor()
        c.execute("DELETE FROM messages WHERE conv_id = ?", (conv_id,))
        c.execute("DELETE FROM conversations WHERE conv_id = ?", (conv_id,))
        self._ensure_conn().commit()

    # ------ 判断日志 ------ #

    def log_judgment(
        self,
        umo: str,
        sender_name: str,
        message: str,
        overall: float,
        relevance: float,
        replyability: float,
        emotional_suitability: float,
        should_reply: bool,
        reason: str,
        conv_id: str | None = None,
        chat_type: str = "",
        chat_id: str = "",
        speaker_target_clarity: float = 0.0,
        privacy_safety_risk: float = 0.0,
        user_intent_clarity: float = 0.0,
        context_completeness: float = 0.0,
        turn_idleness: float = 0.0,
        intervention_naturalness: float = 0.0,
        group_atmosphere_fit: float = 0.0,
    ):
        c = self._cursor()
        c.execute(
            """INSERT INTO judgment_log
               (umo, chat_type, chat_id, conv_id, sender_name, message,
                overall, speaker_target_clarity, privacy_safety_risk,
                relevance, user_intent_clarity, replyability,
                context_completeness, turn_idleness, emotional_suitability,
                intervention_naturalness, group_atmosphere_fit,
                should_reply, reason)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (umo, chat_type, chat_id, conv_id, sender_name, message,
             overall, speaker_target_clarity, privacy_safety_risk,
             relevance, user_intent_clarity, replyability,
             context_completeness, turn_idleness, emotional_suitability,
             intervention_naturalness, group_atmosphere_fit,
             int(should_reply), reason),
        )
        self._ensure_conn().commit()

    def get_recent_judgments(self, umo: str, limit: int = 10) -> list[dict]:
        c = self._cursor()
        rows = c.execute(
            """SELECT id, sender_name, message, overall,
                      speaker_target_clarity, privacy_safety_risk,
                      relevance, user_intent_clarity, replyability,
                      context_completeness, turn_idleness,
                      emotional_suitability, intervention_naturalness,
                      group_atmosphere_fit,
                      should_reply, reason, created_at, chat_type, chat_id, umo
               FROM judgment_log
               WHERE umo = ?
               ORDER BY id DESC LIMIT ?""",
            (umo, limit),
        ).fetchall()
        return [
            {
                "id": r[0], "sender": r[1], "message": r[2], "overall": r[3],
                "speaker_target_clarity": r[4], "privacy_safety_risk": r[5],
                "relevance": r[6], "user_intent_clarity": r[7],
                "replyability": r[8], "context_completeness": r[9],
                "turn_idleness": r[10], "emotional_suitability": r[11],
                "intervention_naturalness": r[12], "group_atmosphere_fit": r[13],
                "should_reply": bool(r[14]), "reason": r[15], "time": r[16],
                "chat_type": r[17], "chat_id": r[18], "umo": r[19],
            }
            for r in rows
        ]

    def get_recent_judgments_all(self, limit: int = 50) -> list[dict]:
        c = self._cursor()
        rows = c.execute(
            """SELECT id, sender_name, message, overall,
                      speaker_target_clarity, privacy_safety_risk,
                      relevance, user_intent_clarity, replyability,
                      context_completeness, turn_idleness,
                      emotional_suitability, intervention_naturalness,
                      group_atmosphere_fit,
                      should_reply, reason, created_at, chat_type, chat_id, umo
               FROM judgment_log
               ORDER BY id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [
            {
                "id": r[0], "sender": r[1], "message": r[2], "overall": r[3],
                "speaker_target_clarity": r[4], "privacy_safety_risk": r[5],
                "relevance": r[6], "user_intent_clarity": r[7],
                "replyability": r[8], "context_completeness": r[9],
                "turn_idleness": r[10], "emotional_suitability": r[11],
                "intervention_naturalness": r[12], "group_atmosphere_fit": r[13],
                "should_reply": bool(r[14]), "reason": r[15], "time": r[16],
                "chat_type": r[17], "chat_id": r[18], "umo": r[19],
            }
            for r in rows
        ]

    def delete_judgment(self, jid: int):
        c = self._cursor()
        c.execute("DELETE FROM judgment_log WHERE id = ?", (jid,))
        self._ensure_conn().commit()

    def delete_judgments_by_ids(self, ids: list[int]):
        c = self._cursor()
        placeholders = ",".join("?" for _ in ids)
        c.execute(f"DELETE FROM judgment_log WHERE id IN ({placeholders})", ids)
        self._ensure_conn().commit()

    def delete_chat_all(self, umo: str):
        c = self._cursor()
        c.execute("DELETE FROM judgment_log WHERE umo = ?", (umo,))
        c.execute("DELETE FROM messages WHERE conv_id = ?", (umo,))
        c.execute("DELETE FROM conversations WHERE conv_id = ?", (umo,))
        self._ensure_conn().commit()

    def clear_all_data(self):
        c = self._cursor()
        c.execute("DELETE FROM judgment_log")
        c.execute("DELETE FROM messages")
        c.execute("DELETE FROM conversations")
        self._ensure_conn().commit()

    def get_distinct_chats(self) -> list[dict]:
        c = self._cursor()
        rows = c.execute(
            """SELECT DISTINCT chat_type, chat_id, umo
               FROM judgment_log
               WHERE chat_type != ''
               ORDER BY chat_type, chat_id"""
        ).fetchall()
        return [
            {"chat_type": r[0], "chat_id": r[1], "umo": r[2]}
            for r in rows
        ]

    # ------ 键值存储 ------ #

    def put(self, key: str, value: Any):
        c = self._cursor()
        c.execute(
            """INSERT INTO kv_store (key, value)
               VALUES (?, ?)
               ON CONFLICT(key) DO UPDATE SET value = excluded.value,
               updated_at = datetime('now', 'localtime')""",
            (key, json.dumps(value, ensure_ascii=False)),
        )
        self._ensure_conn().commit()

    def get(self, key: str, default: Any = None) -> Any:
        c = self._cursor()
        row = c.execute(
            "SELECT value FROM kv_store WHERE key = ?", (key,)
        ).fetchone()
        if row:
            return json.loads(row[0])
        return default

    def delete(self, key: str):
        c = self._cursor()
        c.execute("DELETE FROM kv_store WHERE key = ?", (key,))
        self._ensure_conn().commit()

