"""The router's memory: which model succeeded or failed on which kind of
query. A SQLite table of outcomes (for durability/auditability) plus a
FAISS index over one embedding per unique query, PLUS in-memory mirrors of
both the per-query outcomes and the per-model global success rate.

The in-memory mirrors matter for the latency target (p95 routing overhead
<10ms at 5,000 entries): naively re-querying SQLite for every neighbor's
outcomes and for each model's global AVG(success) on every single routing
call turns into dozens of unindexed-or-not table scans per decision, which
dominates the routing latency at any real scale. Routing reads never touch
SQLite; only writes do.
"""

import os
import json
import time
import uuid
import sqlite3
import numpy as np
import faiss

from app.config import settings as app_settings
from app.routing.config import routing_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS routing_outcomes (
    id TEXT PRIMARY KEY,
    query_id TEXT NOT NULL,
    prompt_text TEXT NOT NULL,
    model_name TEXT NOT NULL,
    quality_score REAL NOT NULL,
    success INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    latency_ms REAL NOT NULL,
    source TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_routing_outcomes_query_id ON routing_outcomes(query_id);",
    "CREATE INDEX IF NOT EXISTS idx_routing_outcomes_model_name ON routing_outcomes(model_name);",
]


class ExperienceStore:
    def __init__(self, data_dir: str = None, pass_mark: float = None, dim: int = 384):
        self.data_dir = data_dir or app_settings.data_dir
        self.pass_mark = pass_mark if pass_mark is not None else routing_settings.pass_mark
        self.dim = dim

        os.makedirs(self.data_dir, exist_ok=True)
        self.db_path = os.path.join(self.data_dir, "routing_experience.db")
        self.index_path = os.path.join(self.data_dir, "routing_experience.index")
        self.ids_path = os.path.join(self.data_dir, "routing_experience_ids.json")

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute(SCHEMA)
        for stmt in INDEXES:
            self.conn.execute(stmt)
        self.conn.commit()

        self.index = faiss.IndexFlatIP(self.dim)
        self.query_ids: list[str] = []   # FAISS position -> query_id
        self._qid_set: set[str] = set()
        self._load_index()

        # In-memory mirrors, rebuilt from SQLite once at startup.
        self._outcomes_by_query: dict[str, dict[str, dict]] = {}
        self._model_success_sum: dict[str, float] = {}
        self._model_count: dict[str, int] = {}
        self._load_memory_mirrors()

    def _load_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.ids_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.ids_path) as f:
                self.query_ids = json.load(f)
            self._qid_set = set(self.query_ids)

    def _save_index(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.ids_path, "w") as f:
            json.dump(self.query_ids, f)

    def _load_memory_mirrors(self):
        cur = self.conn.execute(
            """SELECT query_id, model_name, quality_score, success, cost_usd,
                      latency_ms, created_at
               FROM routing_outcomes ORDER BY created_at ASC"""
        )
        for qid, model, quality, success, cost, latency, _created in cur.fetchall():
            self._outcomes_by_query.setdefault(qid, {})[model] = {
                "quality": quality, "success": bool(success),
                "cost": cost, "latency": latency,
            }
            self._model_success_sum[model] = self._model_success_sum.get(model, 0.0) + success
            self._model_count[model] = self._model_count.get(model, 0) + 1

    # ── writes ──────────────────────────────────────────────────────────────
    def add_outcome(self, query_id: str, embedding, prompt: str, model: str,
                     quality: float, cost: float, latency: float, source: str,
                     _defer_save: bool = False, _defer_commit: bool = False):
        success = 1 if quality >= self.pass_mark else 0
        self.conn.execute(
            """INSERT INTO routing_outcomes
               (id, query_id, prompt_text, model_name, quality_score, success,
                cost_usd, latency_ms, source, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (str(uuid.uuid4()), query_id, prompt, model, float(quality), success,
             float(cost), float(latency), source, time.time())
        )
        if not _defer_commit:
            self.conn.commit()

        # Update in-memory mirrors — this is what routing actually reads.
        self._outcomes_by_query.setdefault(query_id, {})[model] = {
            "quality": float(quality), "success": bool(success),
            "cost": float(cost), "latency": float(latency),
        }
        self._model_success_sum[model] = self._model_success_sum.get(model, 0.0) + success
        self._model_count[model] = self._model_count.get(model, 0) + 1

        if query_id not in self._qid_set:
            vec = np.asarray(embedding, dtype="float32").reshape(1, -1)
            self.index.add(vec)
            self.query_ids.append(query_id)
            self._qid_set.add(query_id)
            if not _defer_save:
                self._save_index()

    def bootstrap_from_dataset(self, rows, commit_every: int = 2000):
        """rows: iterable of dicts with keys query_id, embedding, prompt,
        model, quality, cost, latency, source (defaults to 'offline').
        Defers the FAISS index save until the end, and batches SQLite
        commits every `commit_every` rows instead of one per row — fine for
        one-off live traffic, but committing per-row makes bulk-loading
        hundreds of thousands of rows (e.g. RouterBench) needlessly slow."""
        n = 0
        for r in rows:
            self.add_outcome(
                r["query_id"], r["embedding"], r["prompt"], r["model"],
                r["quality"], r["cost"], r["latency"], r.get("source", "offline"),
                _defer_save=True, _defer_commit=True,
            )
            n += 1
            if n % commit_every == 0:
                self.conn.commit()
        if n:
            self.conn.commit()
            self._save_index()

    # ── reads (in-memory only — no SQLite on the routing hot path) ─────────
    def neighbors(self, embedding, k: int = 30, min_sim: float = 0.0):
        """Returns up to k nearest queries (by embedding), each with its
        similarity and a dict of {model_name: outcome} for every model that
        has an outcome recorded against that query."""
        if self.index.ntotal == 0:
            return []
        vec = np.asarray(embedding, dtype="float32").reshape(1, -1)
        k_eff = min(k, self.index.ntotal)
        sims, positions = self.index.search(vec, k_eff)

        results = []
        for sim, pos in zip(sims[0], positions[0]):
            if pos == -1 or sim < min_sim:
                continue
            qid = self.query_ids[pos]
            results.append({
                "query_id": qid,
                "similarity": float(sim),
                "outcomes": self._outcomes_by_query.get(qid, {}),
            })
        return results

    def global_success_rate(self, model: str) -> float:
        count = self._model_count.get(model, 0)
        if count == 0:
            return 0.5   # no data at all yet — maximally uninformative prior
        return self._model_success_sum[model] / count

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM routing_outcomes").fetchone()[0]

    def unique_query_count(self) -> int:
        return self.index.ntotal


experience_store = ExperienceStore()
