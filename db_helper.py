import sqlite3
import json
from pathlib import Path
from typing import Any

MESSAGE_SCHEMA_DOC = """
发给 LLM 的消息格式 (OpenAI 兼容):

[
  {"role": "system", "content": "人设 + 工具指令"},
  {"role": "user", "content": "用户消息"},
  {"role": "assistant", "content": "助手回复", "tool_calls": [...]},
  {"role": "tool", "content": "工具返回", "tool_call_id": "..."},
  ...
]

本表按此格式拆分为行存储，一条 DB 行 = 一条 message。
"""


class VolitionalDB:
    """Volitional 插件的 SQLite 数据库工具。

    数据文件位于插件数据目录下的 volitional.db。
    路径由外部通过 data_dir 参数传入，符合 AstrBot 插件数据存储规范。
    """

    def __init__(self, data_dir: Path):
        self._db_path = data_dir / "volitional.db"
        self._conn: sqlite3.Connection | None = None

    # ------ 连接管理 ------ #

    def _ensure_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
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
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_msg_conv_seq
            ON messages(conv_id, seq)
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS judgment_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                umo TEXT NOT NULL,
                conv_id TEXT,
                sender_name TEXT,
                message TEXT,
                overall REAL,
                relevance REAL,
                replyability REAL,
                emotional_suitability REAL,
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

    # ------ 对话管理 ------ #

    def create_conversation(self, conv_id: str, umo: str, title: str = ""):
        c = self._cursor()
        c.execute(
            "INSERT INTO conversations (conv_id, umo, title) VALUES (?, ?, ?)",
            (conv_id, umo, title),
        )
        self._ensure_conn().commit()

    def get_conversation(self, conv_id: str) -> dict | None:
        c = self._cursor()
        row = c.execute(
            "SELECT conv_id, umo, title, created_at, updated_at "
            "FROM conversations WHERE conv_id = ?",
            (conv_id,),
        ).fetchone()
        if row:
            return {
                "conv_id": row[0], "umo": row[1], "title": row[2],
                "created_at": row[3], "updated_at": row[4],
            }
        return None

    def list_conversations(self, umo: str) -> list[dict]:
        c = self._cursor()
        rows = c.execute(
            "SELECT conv_id, title, created_at, updated_at "
            "FROM conversations WHERE umo = ? ORDER BY updated_at DESC",
            (umo,),
        ).fetchall()
        return [
            {"conv_id": r[0], "title": r[1], "created_at": r[2], "updated_at": r[3]}
            for r in rows
        ]

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
    ):
        """添加一条消息到对话。

        Args:
            conv_id: 对话 ID。
            role: 角色 (system / user / assistant / tool)。
            content: 消息文本。
            seq: 序号，None 则自动递增。
            tool_calls: assistant 的工具调用列表 (OpenAI 格式)。
            tool_call_id: tool 角色的调用 ID。
            name: 可选的工具名称。
        """
        c = self._cursor()
        if seq is None:
            max_seq = c.execute(
                "SELECT COALESCE(MAX(seq), -1) FROM messages WHERE conv_id = ?",
                (conv_id,),
            ).fetchone()[0]
            seq = max_seq + 1

        c.execute(
            """INSERT INTO messages (conv_id, seq, role, content, tool_calls, tool_call_id, name)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                conv_id, seq, role, content,
                json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None,
                tool_call_id, name,
            ),
        )
        self._ensure_conn().execute(
            "UPDATE conversations SET updated_at = datetime('now', 'localtime') "
            "WHERE conv_id = ?",
            (conv_id,),
        )
        self._ensure_conn().commit()

    def get_messages(self, conv_id: str, limit: int | None = None) -> list[dict]:
        """获取对话的消息列表，按 seq 排序。

        Args:
            conv_id: 对话 ID。
            limit: 返回最近 N 条，None 返回全部。

        Returns:
            list[dict]: 消息列表，格式兼容 OpenAI messages。
        """
        c = self._cursor()
        if limit:
            rows = c.execute(
                """SELECT role, content, tool_calls, tool_call_id, name, created_at
                   FROM messages WHERE conv_id = ?
                   ORDER BY seq DESC LIMIT ?""",
                (conv_id, limit),
            ).fetchall()
            rows = list(reversed(rows))
        else:
            rows = c.execute(
                """SELECT role, content, tool_calls, tool_call_id, name, created_at
                   FROM messages WHERE conv_id = ?
                   ORDER BY seq""",
                (conv_id,),
            ).fetchall()
        result = []
        for r in rows:
            msg = {"role": r[0], "content": r[1], "time": r[5]}
            if r[2]:
                msg["tool_calls"] = json.loads(r[2])
            if r[3]:
                msg["tool_call_id"] = r[3]
            if r[4]:
                msg["name"] = r[4]
            result.append(msg)
        return result

    def get_messages_as_openai_json(self, conv_id: str) -> str:
        """获取对话的完整消息列表，返回 OpenAI 兼容的 JSON 字符串。"""
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
    ):
        c = self._cursor()
        c.execute(
            """INSERT INTO judgment_log
               (umo, conv_id, sender_name, message, overall, relevance, replyability,
                emotional_suitability, should_reply, reason)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (umo, conv_id, sender_name, message, overall, relevance, replyability,
             emotional_suitability, int(should_reply), reason),
        )
        self._ensure_conn().commit()

    def get_recent_judgments(self, umo: str, limit: int = 10) -> list[dict]:
        c = self._cursor()
        rows = c.execute(
            """SELECT sender_name, message, overall, should_reply, reason, created_at
               FROM judgment_log
               WHERE umo = ?
               ORDER BY id DESC LIMIT ?""",
            (umo, limit),
        ).fetchall()
        return [
            {
                "sender": r[0], "message": r[1], "overall": r[2],
                "should_reply": bool(r[3]), "reason": r[4], "time": r[5],
            }
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

    def keys(self) -> list[str]:
        c = self._cursor()
        rows = c.execute("SELECT key FROM kv_store").fetchall()
        return [r[0] for r in rows]
