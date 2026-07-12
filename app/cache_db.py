import sqlite3
import os
import time
from app.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS cache_entries (
    id TEXT PRIMARY KEY,
    prompt_text TEXT NOT NULL,
    response_text TEXT NOT NULL,
    model TEXT NOT NULL,
    system_prompt_hash TEXT NOT NULL,
    temperature REAL NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    hit_count INTEGER DEFAULT 0,
    last_accessed REAL
);
"""

class CacheDB:
    def __init__(self):
        os.makedirs(settings.data_dir, exist_ok=True)
        self.conn = sqlite3.connect(settings.sqlite_path, check_same_thread=False)
        self.conn.execute(SCHEMA)
        self.conn.commit()

    def insert(self, entry: dict):
        self.conn.execute(
            """INSERT INTO cache_entries
               (id, prompt_text, response_text, model, system_prompt_hash, temperature,
                created_at, expires_at, hit_count, last_accessed)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)""",
            (entry["id"], entry["prompt_text"], entry["response_text"], entry["model"],
             entry["system_prompt_hash"], entry["temperature"], entry["created_at"],
             entry["expires_at"], entry["created_at"])
        )
        self.conn.commit()

    def get(self, cache_id: str):
        cur = self.conn.execute("SELECT * FROM cache_entries WHERE id = ?", (cache_id,))
        row = cur.fetchone()
        if row is None:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))

    def register_hit(self, cache_id: str):
        self.conn.execute(
            "UPDATE cache_entries SET hit_count = hit_count + 1, last_accessed = ? WHERE id = ?",
            (time.time(), cache_id)
        )
        self.conn.commit()

    def delete_by_system_prompt(self, system_prompt_hash: str):
        self.conn.execute("DELETE FROM cache_entries WHERE system_prompt_hash = ?", (system_prompt_hash,))
        self.conn.commit()

    def count(self):
        return self.conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0]

cache_db = CacheDB()