import hashlib
import json
import time
import uuid
import numpy as np

from app.embeddings import embedding_service
from app.vector_store import vector_store
from app.cache_db import cache_db
from app.llm_providers import get_provider
from app.ttl_policy import assign_ttl
from app.config import settings
from app.context_engine import (
    build_context_summary, classify_context,
    embed_context, context_similarity,
    context_to_json, context_from_json,
)
from app.context_scorer import (
    score_in_background, score_synchronous, THRESHOLD_FREE, THRESHOLD_MEDIUM
)
from app.template_engine import extract_template, fill_template, check_slot_drift
from app.routing.config import routing_settings
from app.routing.model_registry import model_registry
from app.routing.knn_router import knn_router
from app.routing.verifier import cascade_generate
from app.routing.experience_store import experience_store
from app.routing import judge as routing_judge

# ── Pricing (mock values — replace with real provider pricing later) ──────────
MOCK_PRICE_PER_1K_INPUT  = 0.005
MOCK_PRICE_PER_1K_OUTPUT = 0.015

# Similarity thresholds
PROMPT_THRESHOLD   = settings.similarity_threshold   # 0.92 — how similar must the prompt be
CONTEXT_THRESHOLD  = 0.72   # how similar must the prior context be for Layer 1

AUTO_MODEL = "auto"


def _bucket(system_prompt: str, model: str, temperature: float, quality_target: float = None) -> str:
    if model == AUTO_MODEL:
        # Routed requests bucket on the routing INTENT (quality target), not
        # on any specific underlying model — the whole point of auto-routing
        # is that the same cached answer is reusable regardless of which
        # model happened to produce it.
        raw = f"{system_prompt}|{AUTO_MODEL}|{temperature}|{quality_target}"
    else:
        raw = f"{system_prompt}|{model}|{temperature}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _cost(input_tokens: int, output_tokens: int) -> float:
    return (
        (input_tokens  / 1000) * MOCK_PRICE_PER_1K_INPUT +
        (output_tokens / 1000) * MOCK_PRICE_PER_1K_OUTPUT
    )


def _cost_for(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """Uses the model registry's per-model pricing for routed mock models
    (mock-small/medium/large); falls back to the flat legacy pricing for
    any other model name, so non-routed requests are unaffected."""
    spec = model_registry.get(model_name)
    if spec is not None:
        return model_registry.cost(model_name, input_tokens, output_tokens)
    return _cost(input_tokens, output_tokens)


class CacheResult:
    def __init__(self, text, cache_status, layer=None,
                 similarity=None, context_sim=None,
                 latency_ms=None, cost_usd=0.0, cache_id=None,
                 cost_saved_usd=0.0, routed_model=None, route_reason=None,
                 route_confidence=None, escalated=False,
                 cost_saved_vs_large_usd=0.0):
        self.text            = text
        self.cache_status    = cache_status   # "HIT" or "MISS"
        self.layer           = layer          # 1, 2, or 3 (None for MISS)
        self.similarity      = similarity     # prompt similarity score
        self.context_sim     = context_sim    # context similarity score
        self.latency_ms      = latency_ms
        self.cost_usd        = cost_usd       # $ actually spent (MISS only)
        self.cache_id        = cache_id       # id of the entry served/created
        self.cost_saved_usd  = cost_saved_usd # $ avoided by serving this HIT
                                               # from cache instead of regenerating
        self.routed_model    = routed_model   # actual model that served this request
        self.route_reason    = route_reason   # threshold|utility|fallback_ood|explore|forced
        self.route_confidence = route_confidence
        self.escalated        = escalated     # cascade escalated one tier on this request
        self.cost_saved_vs_large_usd = cost_saved_vs_large_usd  # vs. always using mock-large


def _try_refresh_from_template(cache_id: str, cached_text: str) -> str:
    """Template-slot caching (Feature 1): if this entry has an extracted
    template, do a cheap slot-only re-check instead of serving the possibly
    stale cached text verbatim.

    T0 (small drift)  -> substitute new values directly, no extra generation
    T1 (medium drift)  -> ask only about the changed slots (patch prompt),
                          then substitute
    T2 (large drift / no template) -> too stale to patch safely; serve the
                          cached text unchanged rather than risk stitching
                          together an incoherent answer (full regeneration
                          would belong at Layer 3, not silently mid-HIT)

    This is entirely best-effort: MockProvider doesn't return real JSON, so
    in mock mode this will always fail to parse and fall through to
    returning the original cached text — a real provider is required for
    the refresh itself to do anything.
    """
    try:
        tmpl = cache_db.get_template(cache_id)
        if tmpl is None:
            return cached_text

        provider = get_provider()
        slots_desc = "\n".join(
            f"- {name} ({tmpl['slot_types'].get(name, 'number')}): currently {value}"
            for name, value in tmpl["slot_values"].items()
        )
        refresh_prompt = (
            "The following values were given in a previous answer. Provide "
            "their CURRENT values as a JSON object mapping slot name to new "
            "value (plain numbers only, no symbols).\n\n"
            f"{slots_desc}\n\n"
            'Return ONLY JSON, e.g. {"slot_0": "215.00"}'
        )
        result = provider.generate(
            refresh_prompt, "You are a precise JSON-only data provider.",
            "gpt-4o-mini", 0.0,
        )
        new_values = json.loads(result["text"].strip())

        drift = check_slot_drift(tmpl["slot_values"], new_values, tmpl["slot_types"])
        if drift.tier == "T2":
            return cached_text

        filled = fill_template(tmpl["template_text"], new_values)
        cache_db.update_template_slot_values(cache_id, new_values)
        return filled
    except Exception:
        # Best-effort only — any failure here must never break a cache hit.
        return cached_text


def get_or_generate(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    conversation_history: list = None,
    quality_target: float = None,
    ground_truth_fn=None,
) -> CacheResult:
    """
    Three-layer cache lookup:

    Layer 1 — Exact context match
        Prompt similar AND same bucket AND same context tag
        AND context embeddings are similar.
        → Serve cached answer. Free.

    Layer 2 — Context-free match
        Prompt similar AND same bucket AND the cached answer
        has a low context-sensitivity score (< THRESHOLD_FREE).
        → Safe to serve to anyone. Free.

    Layer 3 — Full generation
        No valid match found. Call the LLM, store with context
        metadata and fire the async sensitivity scorer.

    model="auto" routes through the non-LLM kNN router instead of calling a
    fixed model — see app/routing/. Any explicit model name bypasses routing
    entirely (route_reason="forced"), so this is fully backward compatible.
    `ground_truth_fn` is simulation-only plumbing for the eval harness; the
    live API never passes it.
    """
    start = time.time()
    if conversation_history is None:
        conversation_history = []

    is_auto = (model == AUTO_MODEL)
    if is_auto and quality_target is None:
        quality_target = routing_settings.quality_target

    # ── Build context fingerprint for this request ────────────────────────────
    context_summary = build_context_summary(conversation_history)
    context_tag     = classify_context(context_summary)
    context_vec     = embed_context(context_summary)
    bucket          = _bucket(system_prompt, model, temperature, quality_target if is_auto else None)
    # One embedding per request, computed once here and reused for BOTH the
    # cache lookup (FAISS search below) and routing (knn_router.route below)
    # — never re-embedded.
    query_vec       = embedding_service.embed(prompt)
    now             = time.time()

    # ── FAISS search — get top-k prompt-similar candidates ───────────────────
    candidates = vector_store.search(query_vec, top_k=10)

    def _quality_ok(entry):
        # A cache lookup must never resurface an answer already known (via
        # the judge) to be bad — regardless of how it got cached.
        q = entry.get("quality_score")
        return q is None or q >= routing_settings.pass_mark

    def _hit_route_fields(entry):
        if is_auto:
            return entry.get("model"), entry.get("route_reason"), entry.get("route_confidence"), bool(entry.get("escalated"))
        return entry.get("model"), "forced", None, False

    # ── Layer 1: Exact context match ──────────────────────────────────────────
    for cache_id, prompt_score in candidates:
        if prompt_score < PROMPT_THRESHOLD:
            continue
        entry = cache_db.get(cache_id)
        if entry is None or entry["system_prompt_hash"] != bucket:
            continue
        if entry["expires_at"] < now:
            continue
        if not _quality_ok(entry):
            continue
        if entry["context_tag"] != context_tag:
            continue   # different session type → skip Layer 1, may still pass Layer 2

        # Compare context embeddings if we have one stored
        stored_ctx_json = entry.get("context_embedding_json")
        if stored_ctx_json:
            stored_ctx_vec = context_from_json(stored_ctx_json)
            ctx_sim = context_similarity(context_vec, stored_ctx_vec)
            if ctx_sim >= CONTEXT_THRESHOLD:
                cache_db.register_hit(cache_id)
                served_text = _try_refresh_from_template(cache_id, entry["response_text"])
                r_model, r_reason, r_conf, r_esc = _hit_route_fields(entry)
                return CacheResult(
                    served_text, "HIT", layer=1,
                    similarity=prompt_score, context_sim=ctx_sim,
                    latency_ms=(time.time() - start) * 1000,
                    cache_id=cache_id,
                    cost_saved_usd=entry.get("generation_cost_usd") or 0.0,
                    routed_model=r_model, route_reason=r_reason,
                    route_confidence=r_conf, escalated=r_esc,
                )

    # ── Layer 2: Context-free match ───────────────────────────────────────────
    for cache_id, prompt_score in candidates:
        if prompt_score < PROMPT_THRESHOLD:
            continue
        entry = cache_db.get(cache_id)
        if entry is None or entry["system_prompt_hash"] != bucket:
            continue
        if entry["expires_at"] < now:
            continue
        if not _quality_ok(entry):
            continue

        sensitivity = entry.get("context_sensitivity_score")
        if sensitivity is None:
            # Score not computed yet (scorer is async) — be conservative,
            # treat as medium sensitivity and skip
            continue
        if sensitivity < THRESHOLD_FREE:
            cache_db.register_hit(cache_id)
            served_text = _try_refresh_from_template(cache_id, entry["response_text"])
            r_model, r_reason, r_conf, r_esc = _hit_route_fields(entry)
            return CacheResult(
                served_text, "HIT", layer=2,
                similarity=prompt_score, context_sim=None,
                latency_ms=(time.time() - start) * 1000,
                cache_id=cache_id,
                cost_saved_usd=entry.get("generation_cost_usd") or 0.0,
                routed_model=r_model, route_reason=r_reason,
                route_confidence=r_conf, escalated=r_esc,
            )

    # ── Layer 3: Full generation ──────────────────────────────────────────────
    provider = get_provider()
    route_decision = None
    escalated = False

    if is_auto:
        route_decision = knn_router.route(prompt, query_vec, quality_target=quality_target)
        model_spec = model_registry.get(route_decision.model)
        query_id = str(uuid.uuid4())
        result, served_model, escalated, _verify_reason = cascade_generate(
            provider, prompt, system_prompt, temperature, model_spec,
            registry=model_registry, store=experience_store,
            query_id=query_id, embedding=query_vec,
        )
        cost = _cost_for(served_model, result["input_tokens"], result["output_tokens"])
    else:
        result = provider.generate(prompt, system_prompt, model, temperature)
        cost = _cost(result["input_tokens"], result["output_tokens"])
        served_model = model
        query_id = None

    new_id = str(uuid.uuid4())
    ttl    = assign_ttl(prompt)
    cache_db.insert({
        "id":                       new_id,
        "prompt_text":              prompt,
        "response_text":            result["text"],
        "model":                    served_model,
        "system_prompt_hash":       bucket,
        "temperature":              temperature,
        "created_at":               now,
        "expires_at":               now + ttl,
        "context_tag":              context_tag,
        "context_sensitivity_score": None,   # filled by async scorer below
        "context_embedding_json":   context_to_json(context_vec),
        "generation_cost_usd":      cost,   # what THIS generation cost — every
                                             # future hit on this entry saves
                                             # exactly this much again
        "quality_score":            None,   # filled by the routing judge (auto only)
        "route_reason":             route_decision.reason if route_decision else None,
        "route_confidence":         route_decision.confidence if route_decision else None,
        "escalated":                escalated,
    })
    vector_store.add(new_id, query_vec)

    # Template-slot caching (Feature 1): if the response contains volatile
    # values (prices, percentages, dates, counts), store a reusable
    # template so future hits can substitute fresh values instead of
    # serving this exact response text verbatim forever.
    template_entry = extract_template(prompt, result["text"])
    if template_entry.is_templatable:
        cache_db.upsert_template(
            new_id, template_entry.template_text,
            template_entry.slot_values, template_entry.slot_types,
            template_entry.volatility_score,
        )

    # Score context-sensitivity either synchronously (SYNC_SCORING=true —
    # tests and low-latency environments, where the async version would
    # leave the score unwritten for any immediately-following request) or
    # in the background (production default — zero added latency).
    if settings.sync_scoring:
        score = score_synchronous(prompt, context_summary, result["text"])
        cache_db.update_sensitivity_score(new_id, score)
    else:
        score_in_background(new_id, prompt, context_summary, result["text"])

    cost_saved_vs_large = 0.0
    if is_auto:
        # Router memory: the judge samples a fraction of traffic to label
        # (quality_score -> success/fail), off the hot path unless
        # SYNC_SCORING is set. This is how the router keeps learning.
        routing_judge.maybe_judge(
            query_id=query_id, embedding=query_vec, prompt=prompt,
            answer=result["text"], model=served_model, cost=cost,
            latency=(time.time() - start) * 1000, cache_id=new_id,
            ground_truth_fn=ground_truth_fn,
        )
        large_models = model_registry.by_tier("large")
        if large_models and served_model != large_models[0].name:
            cost_if_large = _cost_for(large_models[0].name, result["input_tokens"], result["output_tokens"])
            cost_saved_vs_large = max(0.0, cost_if_large - cost)

    return CacheResult(
        result["text"], "MISS", layer=3,
        similarity=None, context_sim=None,
        latency_ms=(time.time() - start) * 1000,
        cost_usd=cost,
        cache_id=new_id,
        routed_model=served_model if is_auto else None,
        route_reason=(route_decision.reason if route_decision else "forced"),
        route_confidence=(route_decision.confidence if route_decision else None),
        escalated=escalated,
        cost_saved_vs_large_usd=cost_saved_vs_large,
    )