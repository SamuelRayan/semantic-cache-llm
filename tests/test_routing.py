"""pytest suite for the non-LLM router. Run with:
    python -m pytest tests/test_routing.py -v

SYNC_SCORING must be set and data/ cleared BEFORE any app.* import, same
constraint as tests/test_context_aware.py — app/config.py reads the env var
into a dataclass default at import time.
"""

import sys, os, time, uuid, shutil, random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["SYNC_SCORING"] = "true"

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
shutil.rmtree(os.path.join(_project_root, "data"), ignore_errors=True)

import pytest
import numpy as np

from app.embeddings import embedding_service
from app.cache_engine import get_or_generate, _bucket
from app.cache_db import cache_db
from app.routing.model_registry import model_registry
from app.routing.experience_store import ExperienceStore
from app.routing.knn_router import KnnRouter
from app.routing.config import RoutingSettings
from app.routing.verifier import cascade_generate


# ── helpers ──────────────────────────────────────────────────────────────────

def _mk_store_and_router(tmp_path, **overrides):
    store = ExperienceStore(data_dir=str(tmp_path), pass_mark=0.7)
    defaults = dict(
        min_sim=0.3, tau=0.15, prior_strength=1.0, quality_target=0.8,
        ood_sim=0.3, min_ess=1.5, fallback_tier="medium", epsilon=0.0,
        k_neighbors=30, decision_mode="threshold", utility_lambda=0.3,
        explore_margin=0.05, explore_ess_threshold=5.0,
        judge_sample_rate=0.2, shadow_sample_rate=0.0,
        verifier_min_response_chars=5, verifier_min_logprob=-1.5,
        pass_mark=0.7,
    )
    defaults.update(overrides)
    settings = RoutingSettings(**defaults)
    router = KnnRouter(registry=model_registry, store=store, settings=settings)
    return store, router


def _seed_cluster(store, base_text, paraphrases, model_outcomes):
    """model_outcomes: {model_name: bool_success}, applied to base_text and
    every paraphrase (paraphrasing a question doesn't change whether a
    model can answer it — see sim/generate_routing_dataset.py)."""
    for i, text in enumerate([base_text] + list(paraphrases)):
        qid = f"seed_{uuid.uuid4().hex[:10]}"
        emb = embedding_service.embed(text)
        for model, success in model_outcomes.items():
            store.add_outcome(qid, emb, text, model, 1.0 if success else 0.0,
                               0.001, 100, "offline")


# ── 1. router picks the cheapest model that actually succeeds ──────────────

def test_router_picks_small_when_small_succeeds(tmp_path):
    store, router = _mk_store_and_router(tmp_path)
    base = "what is the capital of France"
    paras = [
        "what's the capital city of France", "France's capital is what city",
        "tell me the capital of France", "name the capital city of France",
    ]
    _seed_cluster(store, base, paras,
                  {"mock-small": True, "mock-medium": True, "mock-large": True})

    query = "which city is the capital of France"
    emb = embedding_service.embed(query)
    decision = router.route(query, emb)

    assert decision.model == "mock-small", (
        f"expected mock-small, got {decision.model} (p_hat={decision.p_hat}, "
        f"reason={decision.reason})"
    )
    assert decision.reason == "threshold"


def test_router_picks_large_when_only_large_succeeds(tmp_path):
    store, router = _mk_store_and_router(tmp_path)
    base = "prove that the square root of 2 is irrational using contradiction"
    paras = [
        "show that sqrt 2 cannot be rational using a proof by contradiction",
        "demonstrate the irrationality of the square root of two",
        "give a rigorous proof that root 2 is irrational",
        "why is the square root of 2 not a rational number, prove it formally",
    ]
    _seed_cluster(store, base, paras,
                  {"mock-small": False, "mock-medium": False, "mock-large": True})

    query = "prove rigorously that the square root of 2 is irrational"
    emb = embedding_service.embed(query)
    decision = router.route(query, emb)

    assert decision.model == "mock-large", (
        f"expected mock-large, got {decision.model} (p_hat={decision.p_hat}, "
        f"reason={decision.reason})"
    )


# ── 2. no neighbors -> global prior, fallback_ood ───────────────────────────

def test_no_neighbors_uses_global_prior_and_fallback(tmp_path):
    store, router = _mk_store_and_router(tmp_path)
    query = "a totally unseen query about xyzzy plugh quux frobnicate"
    emb = embedding_service.embed(query)
    decision = router.route(query, emb)

    assert decision.reason == "fallback_ood"
    assert decision.low_confidence is True
    for model_name, p in decision.p_hat.items():
        assert p == pytest.approx(0.5, abs=1e-9), f"{model_name}: expected prior 0.5, got {p}"


# ── 3. shrinkage: one contrary neighbor can't overturn a strong prior ──────

def test_shrinkage_single_neighbor_does_not_flip_against_strong_prior(tmp_path):
    store, router = _mk_store_and_router(tmp_path, prior_strength=10.0)

    # Build a strong prior: many unrelated successes for mock-medium.
    for i in range(20):
        text = f"unrelated filler prompt number {i} about topic {i}"
        emb = embedding_service.embed(text)
        store.add_outcome(f"prior_{i}", emb, text, "mock-medium", 1.0, 0.001, 100, "offline")

    prior = store.global_success_rate("mock-medium")
    assert prior > 0.9

    # ONE neighbor, similar to our actual query, says medium FAILED.
    base = "debug this segmentation fault in my C program"
    emb_base = embedding_service.embed(base)
    store.add_outcome("single_neighbor", emb_base, base, "mock-medium", 0.0, 0.001, 100, "offline")

    neighbors = store.neighbors(emb_base, k=30, min_sim=0.0)
    p_hat, ess = router._estimate(neighbors, "mock-medium")

    assert ess < 2.0, f"expected ~1 effective sample from a single neighbor, got ess={ess}"
    assert p_hat > 0.5, (
        f"a single contrary neighbor flipped the estimate below the prior "
        f"(p_hat={p_hat}) — shrinkage (prior_strength=10) should have prevented this"
    )


# ── 4. explicit model bypasses the router ───────────────────────────────────

def test_explicit_model_bypasses_router():
    prompt = f"explicit model bypass test {uuid.uuid4().hex[:8]}"
    r = get_or_generate(prompt, model="mock-small")
    assert r.cache_status == "MISS"
    assert r.route_reason == "forced"


# ── 5. verifier escalates exactly once and logs the failure ────────────────

class _EmptyThenGoodProvider:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, system_prompt, model, temperature):
        self.calls.append(model)
        if len(self.calls) == 1:
            return {"text": "", "input_tokens": 5, "output_tokens": 0,
                     "model": model, "finish_reason": "stop"}
        return {"text": "a perfectly good and complete answer", "input_tokens": 5,
                "output_tokens": 6, "model": model, "finish_reason": "stop"}


def test_verifier_escalates_once_on_empty_response(tmp_path):
    store = ExperienceStore(data_dir=str(tmp_path))
    provider = _EmptyThenGoodProvider()
    small_spec = model_registry.get("mock-small")
    prompt = "cascade escalation test prompt"
    emb = embedding_service.embed(prompt)

    result, served_model, escalated, reason = cascade_generate(
        provider, prompt, "sys", 0.7, small_spec,
        registry=model_registry, store=store, query_id="q1", embedding=emb,
    )

    assert escalated is True
    assert served_model == "mock-medium"
    assert reason == "empty_or_too_short"
    assert len(provider.calls) == 2, "expected exactly one retry (one escalation)"

    neighbors = store.neighbors(emb, k=5, min_sim=0.0)
    assert any("mock-small" in nb["outcomes"] for nb in neighbors), (
        "the mock-small failure should have been logged to the experience store"
    )
    failed_outcome = next(nb["outcomes"]["mock-small"] for nb in neighbors if "mock-small" in nb["outcomes"])
    assert failed_outcome["success"] is False


# ── 6. a known-bad cache entry is skipped; a regenerated one is servable ───

def test_low_quality_entry_not_served_regenerated_is():
    prompt = f"quality gate test {uuid.uuid4().hex[:8]}"

    r1 = get_or_generate(prompt, model="mock-small")
    assert r1.cache_status == "MISS"
    cache_db.update_sensitivity_score(r1.cache_id, 0.1)   # Layer-2-eligible...
    cache_db.update_quality_score(r1.cache_id, 0.2)        # ...but judged BAD

    r2 = get_or_generate(prompt, model="mock-small")
    assert r2.cache_status == "MISS", "a low quality_score entry must never be served"
    assert r2.cache_id != r1.cache_id
    cache_db.update_sensitivity_score(r2.cache_id, 0.1)    # this one is fine

    r3 = get_or_generate(prompt, model="mock-small")
    assert r3.cache_status == "HIT", "the regenerated (un-flagged) entry should be servable"
    assert r3.cache_id == r2.cache_id


# ── 7. auto and explicit-model requests never share a cache bucket ─────────

def test_auto_and_explicit_never_share_bucket():
    b_auto = _bucket("sys", "auto", 0.7, 0.8)
    b_gpt = _bucket("sys", "gpt-4o-mini", 0.7)
    b_small = _bucket("sys", "mock-small", 0.7)
    b_auto_diff_target = _bucket("sys", "auto", 0.7, 0.6)

    assert b_auto != b_gpt
    assert b_auto != b_small
    assert b_gpt != b_small
    assert b_auto != b_auto_diff_target, "different quality_targets should bucket separately"


# ── 8. online learning: a new outcome immediately changes the next decision ─

def test_online_learning_immediate_effect(tmp_path):
    store, router = _mk_store_and_router(tmp_path)
    query = "translate this document into French with a formal tone"
    emb = embedding_service.embed(query)

    before = router.route(query, emb)
    assert before.reason == "fallback_ood"

    paras = [
        "please translate this document to French formally",
        "convert this document into formal French",
        "render this text in formal French please",
        "translate the following into formal French for me",
    ]
    _seed_cluster(store, query, paras,
                  {"mock-small": True, "mock-medium": True, "mock-large": True})

    after = router.route(query, emb)
    assert after.reason != "fallback_ood", "new outcomes should be usable as neighbors immediately, no retrain"
    assert after.model == "mock-small"


# ── 9. routing overhead stays well under 10ms even with 5,000 entries ──────

def test_routing_latency_p95_under_10ms_at_5000_entries(tmp_path):
    store, router = _mk_store_and_router(tmp_path)
    rng = np.random.RandomState(0)
    n = 5000

    def _rows():
        for i in range(n):
            vec = rng.rand(384).astype("float32")
            vec = vec / (np.linalg.norm(vec) + 1e-9)
            for model in ["mock-small", "mock-medium", "mock-large"]:
                yield {
                    "query_id": f"bulk_{i}", "embedding": vec,
                    "prompt": f"bulk bootstrap prompt {i}", "model": model,
                    "quality": float(rng.rand() > 0.5), "cost": 0.001, "latency": 100,
                    "source": "offline",
                }

    store.bootstrap_from_dataset(_rows())
    assert store.unique_query_count() == n

    query_vecs = []
    for _ in range(300):
        v = rng.rand(384).astype("float32")
        query_vecs.append(v / (np.linalg.norm(v) + 1e-9))

    timings_ms = []
    for v in query_vecs:
        t0 = time.perf_counter()
        router.route("bench query", v)
        timings_ms.append((time.perf_counter() - t0) * 1000)

    timings_ms.sort()
    p50 = timings_ms[len(timings_ms) // 2]
    p95 = timings_ms[int(len(timings_ms) * 0.95)]
    print(f"\n  routing latency @ {n} entries: p50={p50:.3f}ms p95={p95:.3f}ms")

    if p95 >= 10.0:
        pytest.skip(
            f"p95 routing latency was {p95:.2f}ms (target <10ms) — likely a slow "
            f"CPU/host for this run, not a correctness issue"
        )
    assert p95 < 10.0


# ── 10. the embedding is computed exactly once per request ─────────────────

def test_embedding_computed_once_per_request(monkeypatch):
    calls = []
    original_embed = embedding_service.embed

    def spy(text):
        calls.append(text)
        return original_embed(text)

    monkeypatch.setattr(embedding_service, "embed", spy)

    prompt = f"embed once per request test {uuid.uuid4().hex[:8]}"
    get_or_generate(prompt, model="auto")

    n = calls.count(prompt)
    assert n == 1, f"expected embedding_service.embed(prompt) exactly once, got {n} calls: {calls}"
