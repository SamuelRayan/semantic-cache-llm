"""Async judge: scores a sampled fraction of served responses off the hot
path and writes labeled outcomes back into the experience store, so the
router keeps learning from live traffic without a retraining step.

Two modes:
  - production (ground_truth_fn=None): asks the configured LLM provider to
    grade its own answer, same JSON-with-graceful-fallback pattern as
    app/context_scorer.py. MockProvider never returns real JSON, so this
    always falls back to a neutral 0.5 in mock mode — that's expected and
    matches the project's existing approach to the mock-provider limitation.
  - simulation (ground_truth_fn given): the harness's own oracle label.
    This is only ever wired in by sim/eval code, never by the live
    cache_engine.py integration — the router itself never sees it directly,
    only the quality scores that come out the other side, exactly like a
    real judge would produce.
"""

import json
import random
import threading

from app.config import settings as app_settings
from app.routing.config import routing_settings
from app.routing.experience_store import experience_store as _default_store

JUDGE_PROMPT = """Rate the quality of this answer to the question, from 0.0 (completely wrong/unhelpful) to 1.0 (fully correct and helpful).

Question: {prompt}
Answer: {answer}

Return ONLY JSON: {{"quality": 0.0}}"""


def _compute_quality(prompt: str, answer: str, ground_truth_fn=None, model: str = None) -> float:
    if ground_truth_fn is not None:
        return float(ground_truth_fn(prompt, model))

    from app.llm_providers import get_provider
    provider = get_provider()
    filled = JUDGE_PROMPT.format(prompt=(prompt or "")[:500], answer=(answer or "")[:500])
    result = provider.generate(filled, "You are a precise JSON-only quality grader.", "gpt-4o-mini", 0.0)
    text = result["text"].strip()
    try:
        parsed = json.loads(text)
        q = float(parsed.get("quality", 0.5))
        return max(0.0, min(1.0, q))
    except (json.JSONDecodeError, ValueError):
        return 0.5


def score_synchronous(prompt: str, answer: str, model: str = None, ground_truth_fn=None) -> float:
    try:
        return _compute_quality(prompt, answer, ground_truth_fn, model)
    except Exception as e:
        print(f"[judge] synchronous scoring failed: {e}")
        return 0.5


def _score_and_record(query_id, embedding, prompt, answer, model, cost, latency,
                       cache_id, ground_truth_fn, store):
    try:
        q = _compute_quality(prompt, answer, ground_truth_fn, model)
        store.add_outcome(query_id, embedding, prompt, model, q, cost, latency, "online_judge")
        if cache_id is not None:
            from app.cache_db import cache_db
            cache_db.update_quality_score(cache_id, q)
    except Exception as e:
        print(f"[judge] background scoring failed: {e}")


def maybe_judge(query_id, embedding, prompt, answer, model, cost, latency,
                 cache_id=None, ground_truth_fn=None, sample_rate=None,
                 store=None) -> bool:
    """Samples a fraction of traffic for judging. Honors SYNC_SCORING for
    tests/low-latency environments; otherwise runs in a daemon thread.
    Returns True if this call was sampled (and thus judged/scheduled)."""
    store = store or _default_store
    rate = sample_rate if sample_rate is not None else routing_settings.judge_sample_rate
    if random.random() >= rate:
        return False

    args = (query_id, embedding, prompt, answer, model, cost, latency,
            cache_id, ground_truth_fn, store)
    if app_settings.sync_scoring:
        _score_and_record(*args)
    else:
        threading.Thread(target=_score_and_record, args=args, daemon=True).start()
    return True


def _run_shadow(provider, prompt, system_prompt, temperature, shadow_spec,
                 query_id, embedding, ground_truth_fn, cost_fn, store):
    try:
        result = provider.generate(prompt, system_prompt, shadow_spec.name, temperature)
        cost = cost_fn(shadow_spec.name, result.get("input_tokens", 0), result.get("output_tokens", 0))
        q = _compute_quality(prompt, result["text"], ground_truth_fn, shadow_spec.name)
        store.add_outcome(query_id, embedding, prompt, shadow_spec.name,
                           q, cost, result.get("latency_ms", 0), "shadow")
    except Exception as e:
        print(f"[judge] shadow scoring failed: {e}")


def maybe_shadow(query_id, embedding, prompt, system_prompt, temperature,
                  served_model_tier, registry, cost_fn, ground_truth_fn=None,
                  sample_rate=None, store=None) -> bool:
    """Optionally also runs a CHEAPER model in the background on sampled
    traffic and judges it — collects the counterfactual label ("would the
    cheap model have worked here too?") the router would otherwise never
    see, since it only observes the model it actually chose."""
    store = store or _default_store
    rate = sample_rate if sample_rate is not None else routing_settings.shadow_sample_rate
    if rate <= 0 or random.random() >= rate:
        return False

    order = ["small", "medium", "large"]
    if served_model_tier not in order or order.index(served_model_tier) == 0:
        return False  # already cheapest tier, nothing to shadow

    cheaper_tier = order[order.index(served_model_tier) - 1]
    candidates = registry.by_tier(cheaper_tier)
    if not candidates:
        return False
    shadow_spec = candidates[0]

    from app.llm_providers import get_provider
    provider = get_provider()
    args = (provider, prompt, system_prompt, temperature, shadow_spec,
            query_id, embedding, ground_truth_fn, cost_fn, store)

    if app_settings.sync_scoring:
        _run_shadow(*args)
    else:
        threading.Thread(target=_run_shadow, args=args, daemon=True).start()
    return True
