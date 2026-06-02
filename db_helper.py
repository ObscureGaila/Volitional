import sqlite3
import json
from pathlib import Path
from typing import Any


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
            CREATE TABLE IF NOT EXISTS judgment_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                umo TEXT NOT NULL,
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
    ):
        c = self._cursor()
        c.execute(
            """INSERT INTO judgment_log
               (umo, sender_name, message, overall, relevance, replyability,
                emotional_suitability, should_reply, reason)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (umo, sender_name, message, overall, relevance, replyability,
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
