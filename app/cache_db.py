import sqlite3
import os
import json
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
    last_accessed REAL,
    context_tag TEXT DEFAULT 'cold_start',
    context_sensitivity_score REAL DEFAULT NULL,
    context_embedding_json TEXT DEFAULT NULL
);
"""

SCHEMA_TEMPLATES = """
CREATE TABLE IF NOT EXISTS template_entries (
    cache_entry_id TEXT PRIMARY KEY,
    template_text TEXT NOT NULL,
    slot_values_json TEXT NOT NULL,
    slot_types_json TEXT NOT NULL,
    volatility_score REAL NOT NULL,
    last_slot_refresh REAL NOT NULL,
    FOREIGN KEY (cache_entry_id) REFERENCES cache_entries(id)
);
"""

class CacheDB:
    def __init__(self):
        os.makedirs(settings.data_dir, exist_ok=True)
        self.conn = sqlite3.connect(settings.sqlite_path, check_same_thread=False)
        self.conn.execute(SCHEMA)
        self.conn.execute(SCHEMA_TEMPLATES)
        self._migrate()
        self.conn.commit()

    def _migrate(self):
        """Add new columns to existing databases that were created before
        this schema version. Safe to run every startup — ALTER TABLE IF NOT
        EXISTS isn't valid SQLite syntax, so we check the column list first."""
        cur = self.conn.execute("PRAGMA table_info(cache_entries)")
        existing_columns = {row[1] for row in cur.fetchall()}
        migrations = [
            ("context_tag",                  "TEXT DEFAULT 'cold_start'"),
            ("context_sensitivity_score",    "REAL DEFAULT NULL"),
            ("context_embedding_json",       "TEXT DEFAULT NULL"),
            ("generation_cost_usd",          "REAL DEFAULT 0.0"),
            ("quality_score",                "REAL DEFAULT NULL"),
            ("route_reason",                 "TEXT DEFAULT NULL"),
            ("route_confidence",             "REAL DEFAULT NULL"),
            ("escalated",                    "INTEGER DEFAULT 0"),
        ]
        for col_name, col_def in migrations:
            if col_name not in existing_columns:
                self.conn.execute(
                    f"ALTER TABLE cache_entries ADD COLUMN {col_name} {col_def}"
                )
        self.conn.commit()

    def insert(self, entry: dict):
        self.conn.execute(
            """INSERT INTO cache_entries
               (id, prompt_text, response_text, model, system_prompt_hash,
                temperature, created_at, expires_at, hit_count, last_accessed,
                context_tag, context_sensitivity_score, context_embedding_json,
                generation_cost_usd, quality_score, route_reason,
                route_confidence, escalated)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (entry["id"], entry["prompt_text"], entry["response_text"],
             entry["model"], entry["system_prompt_hash"], entry["temperature"],
             entry["created_at"], entry["expires_at"], entry["created_at"],
             entry.get("context_tag", "cold_start"),
             entry.get("context_sensitivity_score", None),
             entry.get("context_embedding_json", None),
             entry.get("generation_cost_usd", 0.0),
             entry.get("quality_score", None),
             entry.get("route_reason", None),
             entry.get("route_confidence", None),
             1 if entry.get("escalated") else 0)
        )
        self.conn.commit()

    def update_quality_score(self, cache_id: str, score: float):
        self.conn.execute(
            "UPDATE cache_entries SET quality_score = ? WHERE id = ?",
            (score, cache_id)
        )
        self.conn.commit()

    def get(self, cache_id: str):
        cur = self.conn.execute(
            "SELECT * FROM cache_entries WHERE id = ?", (cache_id,)
        )
        row = cur.fetchone()
        if row is None:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))

    def update_sensitivity_score(self, cache_id: str, score: float):
        self.conn.execute(
            "UPDATE cache_entries SET context_sensitivity_score = ? WHERE id = ?",
            (score, cache_id)
        )
        self.conn.commit()

    def register_hit(self, cache_id: str):
        self.conn.execute(
            """UPDATE cache_entries
               SET hit_count = hit_count + 1, last_accessed = ?
               WHERE id = ?""",
            (time.time(), cache_id)
        )
        self.conn.commit()

    def delete_by_system_prompt(self, system_prompt_hash: str):
        self.conn.execute(
            "DELETE FROM cache_entries WHERE system_prompt_hash = ?",
            (system_prompt_hash,)
        )
        self.conn.commit()

    def count(self):
        return self.conn.execute(
            "SELECT COUNT(*) FROM cache_entries"
        ).fetchone()[0]

    def get_savings_stats(self):
        """Total money saved so far: every cache hit avoided paying the
        entry's original generation cost again. hit_count already tracks
        how many times each entry was served from cache, so
        generation_cost_usd * hit_count is that entry's running total."""
        row = self.conn.execute(
            """SELECT
                 COALESCE(SUM(generation_cost_usd * hit_count), 0.0) AS total_saved_usd,
                 COALESCE(SUM(hit_count), 0)                          AS total_hits,
                 COALESCE(SUM(generation_cost_usd), 0.0)               AS total_generation_cost_usd,
                 COUNT(*)                                              AS total_entries
               FROM cache_entries"""
        ).fetchone()
        total_saved_usd, total_hits, total_generation_cost_usd, total_entries = row
        return {
            "total_saved_usd": total_saved_usd,
            "total_hits": total_hits,
            "avg_saved_per_hit_usd": (total_saved_usd / total_hits) if total_hits else 0.0,
            "total_generation_cost_usd": total_generation_cost_usd,
            "total_entries": total_entries,
        }

    def upsert_template(self, cache_entry_id: str, template_text: str,
                         slot_values: dict, slot_types: dict,
                         volatility_score: float):
        now = time.time()
        self.conn.execute(
            """INSERT INTO template_entries
               (cache_entry_id, template_text, slot_values_json,
                slot_types_json, volatility_score, last_slot_refresh)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(cache_entry_id) DO UPDATE SET
                 template_text=excluded.template_text,
                 slot_values_json=excluded.slot_values_json,
                 slot_types_json=excluded.slot_types_json,
                 volatility_score=excluded.volatility_score,
                 last_slot_refresh=excluded.last_slot_refresh""",
            (cache_entry_id, template_text, json.dumps(slot_values),
             json.dumps(slot_types), volatility_score, now)
        )
        self.conn.commit()

    def get_template(self, cache_entry_id: str):
        cur = self.conn.execute(
            "SELECT * FROM template_entries WHERE cache_entry_id = ?",
            (cache_entry_id,)
        )
        row = cur.fetchone()
        if row is None:
            return None
        cols = [d[0] for d in cur.description]
        entry = dict(zip(cols, row))
        entry["slot_values"] = json.loads(entry["slot_values_json"])
        entry["slot_types"] = json.loads(entry["slot_types_json"])
        return entry

    def update_template_slot_values(self, cache_entry_id: str, slot_values: dict):
        self.conn.execute(
            """UPDATE template_entries
               SET slot_values_json = ?, last_slot_refresh = ?
               WHERE cache_entry_id = ?""",
            (json.dumps(slot_values), time.time(), cache_entry_id)
        )
        self.conn.commit()

    def get_all_with_context(self):
        """Used by the three-layer lookup to fetch candidates that have
        a context_tag and context_embedding stored."""
        cur = self.conn.execute(
            """SELECT id, context_tag, context_sensitivity_score,
                      context_embedding_json
               FROM cache_entries
               WHERE expires_at > ?""",
            (time.time(),)
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

cache_db = CacheDB()