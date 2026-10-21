"""SQLite cache so re-runs never reprocess a clip already seen."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    size INTEGER,
    mtime REAL,
    duration REAL,
    width INTEGER,
    height INTEGER,
    created_at TEXT,
    has_audio INTEGER,
    language TEXT,
    transcript TEXT,
    is_content REAL,
    channel TEXT,
    confidence REAL,
    reason TEXT,
    suggested_title TEXT,
    classified INTEGER DEFAULT 0,
    error TEXT,
    processed_at TEXT
);
"""


class Cache:
    def __init__(self, db_path: Path):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def get(self, video_id: str) -> sqlite3.Row | None:
        cur = self.conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,))
        return cur.fetchone()

    def upsert(self, row: dict) -> None:
        cols = ", ".join(row.keys())
        placeholders = ", ".join("?" for _ in row)
        updates = ", ".join(f"{k}=excluded.{k}" for k in row if k != "id")
        self.conn.execute(
            f"INSERT INTO videos ({cols}) VALUES ({placeholders}) "
            f"ON CONFLICT(id) DO UPDATE SET {updates}",
            list(row.values()),
        )
        self.conn.commit()

    def all(self) -> list[sqlite3.Row]:
        cur = self.conn.execute("SELECT * FROM videos ORDER BY is_content DESC")
        return cur.fetchall()

    def close(self) -> None:
        self.conn.close()
