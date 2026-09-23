"""pytest suite for the real-data (RouterBench) routing validation track.
Run with:
    python -m pytest tests/test_routing_real.py -v

Only test 5 (and the shared fixture behind tests 4/5) actually runs the real
eval script, and only on a small --max-categories/--max-prompts-per-category
subset with isolated --real-dir/--results-dir — never the full 36k-row run,
and never the production data/ directory.
"""

import sys, os, subprocess, time
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "sim", "data", "routerbench_routing_dataset.jsonl")

import eval.run_routing_eval_real as real_eval

pytestmark = pytest.mark.skipif(
    not os.path.exists(DATA_PATH),
    reason=f"{DATA_PATH} not found — run load_routerbench.py first",
)


# ── 1. dataset loads with expected columns ──────────────────────────────────

def test_dataset_loads_with_expected_columns():
    rows = real_eval.load_rows(DATA_PATH)
    assert len(rows) > 0

    expected_cols = {"prompt", "eval_name", "sample_id", "model",
                      "quality_score", "success", "cost_usd", "tier"}
    assert expected_cols <= set(rows[0].keys())

    models = {r["model"] for r in rows}
    assert len(models) >= 5, f"expected several real models, got {models}"
    tiers = {r["tier"] for r in rows}
    assert tiers <= {"small", "medium", "large"}


# ── 2. stratified split preserves per-category proportions ─────────────────

def test_stratified_split_preserves_category_proportions():
    rows = real_eval.load_rows(DATA_PATH)
    prompts_meta, outcome_lookup, model_names = real_eval.build_prompt_meta_and_lookup(rows)
    prompts_meta = real_eval.filter_full_coverage(prompts_meta, outcome_lookup, model_names)

    train, test = real_eval.stratified_split(prompts_meta, test_frac=0.2, seed=42)

    train_by_cat = Counter(p["eval_name"] for p in train)
    test_by_cat = Counter(p["eval_name"] for p in test)

    # every category present overall must be present on BOTH sides (that's
    # the whole point of stratifying) unless it's a singleton category.
    all_cats = set(train_by_cat) | set(test_by_cat)
    checked = 0
    for cat in all_cats:
        total = train_by_cat[cat] + test_by_cat[cat]
        if total < 5:
            continue   # too small for a meaningful ratio check
        frac = test_by_cat[cat] / total
        assert abs(frac - 0.2) < 0.15, (
            f"category {cat!r}: test fraction {frac:.2f} is too far from the "
            f"target 0.2 (train={train_by_cat[cat]}, test={test_by_cat[cat]})"
        )
        checked += 1
    assert checked > 10, "expected to actually check a meaningful number of categories"


# ── 3. embeddings are cached and not recomputed on a second run ────────────

def test_embeddings_cached_and_not_recomputed(tmp_path, monkeypatch):
    monkeypatch.setattr(real_eval, "REAL_DIR", str(tmp_path))

    prompts_meta = [
        {"sample_id": f"s{i}", "prompt": f"a short test prompt number {i}", "eval_name": "cat"}
        for i in range(8)
    ]

    from app.embeddings import embedding_service
    calls = {"n": 0}
    original_encode = embedding_service.model.encode

    def spy_encode(*args, **kwargs):
        calls["n"] += 1
        return original_encode(*args, **kwargs)

    monkeypatch.setattr(embedding_service.model, "encode", spy_encode)

    embs1 = real_eval.get_or_build_embeddings(prompts_meta)
    assert calls["n"] == 1, "expected exactly one encode() batch call on first build"
    assert os.path.exists(os.path.join(str(tmp_path), "embeddings.npy"))

    embs2 = real_eval.get_or_build_embeddings(prompts_meta)
    assert calls["n"] == 1, "second call should have used the on-disk cache, not re-embedded"
    assert np.array_equal(embs1, embs2)


# ── 4 & 5. isolated smoke run: production cache untouched + runs end-to-end ─

@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    real_dir = str(tmp_path_factory.mktemp("routing_real"))
    results_dir = str(tmp_path_factory.mktemp("results"))

    prod_cache_db = os.path.join(PROJECT_ROOT, "data", "cache.db")
    prod_cache_index = os.path.join(PROJECT_ROOT, "data", "cache.index")
    mtime_before = {
        prod_cache_db: os.path.getmtime(prod_cache_db) if os.path.exists(prod_cache_db) else None,
        prod_cache_index: os.path.getmtime(prod_cache_index) if os.path.exists(prod_cache_index) else None,
    }

    t0 = time.time()
    result = subprocess.run(
        [
            sys.executable, os.path.join(PROJECT_ROOT, "eval", "run_routing_eval_real.py"),
            "--max-categories", "2", "--max-prompts-per-category", "80",
            "--real-dir", real_dir, "--results-dir", results_dir,
        ],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=300,
    )
    elapsed = time.time() - t0

    return {
        "result": result, "elapsed": elapsed,
        "real_dir": real_dir, "results_dir": results_dir,
        "prod_cache_db": prod_cache_db, "prod_cache_index": prod_cache_index,
        "mtime_before": mtime_before,
    }


def test_production_cache_untouched_by_real_track(smoke_run):
    assert smoke_run["result"].returncode == 0, (
        f"smoke run failed (needed for this check):\n{smoke_run['result'].stderr[-3000:]}"
    )
    for path in (smoke_run["prod_cache_db"], smoke_run["prod_cache_index"]):
        before = smoke_run["mtime_before"][path]
        after = os.path.getmtime(path) if os.path.exists(path) else None
        assert after == before, f"{path} mtime changed — the real-data track touched the production cache!"

    assert os.path.isdir(smoke_run["real_dir"])
    assert os.path.exists(os.path.join(smoke_run["real_dir"], "models_real.yaml"))
    assert os.path.isdir(os.path.join(smoke_run["real_dir"], "experience_store"))


def test_smoke_run_end_to_end(smoke_run):
    result, elapsed = smoke_run["result"], smoke_run["elapsed"]
    assert result.returncode == 0, f"stdout:\n{result.stdout[-2000:]}\nstderr:\n{result.stderr[-3000:]}"
    assert elapsed < 300, f"smoke run took {elapsed:.0f}s, expected well under 300s for a 2-category sample"

    results_dir = smoke_run["results_dir"]
    assert os.path.exists(os.path.join(results_dir, "real_summary.csv"))
    assert os.path.exists(os.path.join(results_dir, "real_per_category.csv"))
    assert os.path.exists(os.path.join(results_dir, "real_cost_quality_curve.png"))
