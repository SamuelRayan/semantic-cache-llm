import itertools
import time
from fastapi import FastAPI, Response
from pydantic import BaseModel
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.cache_engine import get_or_generate, _bucket
from app.cache_db import cache_db
from app.vector_store import vector_store
from app.metrics import (
    CACHE_HITS, CACHE_MISSES, REQUEST_LATENCY, COST_SAVED, registry,
    ROUTED_REQUESTS, ESCALATIONS, ROUTING_LATENCY, COST_SAVED_VS_LARGE,
)

app = FastAPI(title="Semantic Cache Proxy — Context-Aware Edition")


class ChatRequest(BaseModel):
    prompt: str
    system_prompt: str = "You are a helpful assistant."
    model: str = "gpt-4o-mini"          # pass "auto" to route non-LLM
    temperature: float = 0.7
    conversation_history: list = []   # NEW — list of {"role":..,"content":..} dicts
    quality_target: float = None      # only used when model="auto"


class InvalidateRequest(BaseModel):
    system_prompt: str
    model: str = "gpt-4o-mini"
    temperature: float = 0.7


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest, response: Response):
    start = time.time()
    result = get_or_generate(
        req.prompt,
        req.system_prompt,
        req.model,
        req.temperature,
        req.conversation_history,   # NEW
        req.quality_target,         # NEW — only meaningful when model="auto"
    )
    duration = time.time() - start

    response.headers["X-Cache-Status"]      = result.cache_status
    response.headers["X-Cache-Layer"]       = str(result.layer or "")
    if result.similarity is not None:
        response.headers["X-Cache-Similarity"] = f"{result.similarity:.4f}"
    if result.context_sim is not None:
        response.headers["X-Context-Similarity"] = f"{result.context_sim:.4f}"
    if result.cache_status == "HIT":
        response.headers["X-Cost-Saved-USD"] = f"{result.cost_saved_usd:.6f}"
    if result.routed_model is not None:
        response.headers["X-Routed-Model"] = result.routed_model
    if result.route_reason is not None:
        response.headers["X-Route-Reason"] = result.route_reason
    if result.route_confidence is not None:
        response.headers["X-Route-Confidence"] = f"{result.route_confidence:.4f}"
    response.headers["X-Escalated"] = "true" if result.escalated else "false"

    if result.cache_status == "HIT":
        CACHE_HITS.inc()
        COST_SAVED.inc(result.cost_saved_usd)
    else:
        CACHE_MISSES.inc()

    REQUEST_LATENCY.labels(cache_status=result.cache_status).observe(duration)

    if req.model == "auto" and result.route_reason is not None:
        ROUTED_REQUESTS.labels(model=result.routed_model, reason=result.route_reason).inc()
        if result.escalated:
            ESCALATIONS.labels(from_model="?", to_model=result.routed_model).inc()
        if result.cache_status == "MISS":
            COST_SAVED_VS_LARGE.inc(result.cost_saved_vs_large_usd)

    return {
        "id":              "chatcmpl-mock",
        "model":           req.model,
        "routed_model":    result.routed_model,
        "route_reason":    result.route_reason,
        "route_confidence": result.route_confidence,
        "escalated":       result.escalated,
        "cache_status":    result.cache_status,
        "cache_layer":     result.layer,
        "cache_similarity": result.similarity,
        "context_similarity": result.context_sim,
        "cost_usd":        round(result.cost_usd, 6),
        "cost_saved_usd":  round(result.cost_saved_usd, 6),
        "choices": [{"message": {"role": "assistant", "content": result.text}}],
    }


@app.post("/v1/cache/invalidate")
def invalidate(req: InvalidateRequest):
    b = _bucket(req.system_prompt, req.model, req.temperature)
    cache_db.delete_by_system_prompt(b)
    return {"status": "invalidated", "bucket": b}


@app.get("/v1/cache/stats")
def stats():
    return {"total_entries": cache_db.count()}


@app.get("/v1/cache/savings")
def savings():
    """Running total of money saved by serving cache hits instead of
    calling the LLM again — each entry's generation_cost_usd (what it
    actually cost to produce, using the same mock pricing as the live
    request path) times how many times it's been served from cache."""
    s = cache_db.get_savings_stats()
    return {
        "total_saved_usd":            round(s["total_saved_usd"], 4),
        "total_cache_hits":           s["total_hits"],
        "avg_saved_per_hit_usd":      round(s["avg_saved_per_hit_usd"], 6),
        "total_generation_cost_usd":  round(s["total_generation_cost_usd"], 4),
        "total_entries":              s["total_entries"],
    }


@app.get("/v1/cache/threshold-report")
def threshold_report():
    thresholds = [0.85, 0.90, 0.92, 0.95, 0.98]
    n = vector_store.index.ntotal
    if n < 2:
        return {"message": "Not enough entries yet."}
    all_vecs = vector_store.index.reconstruct_n(0, n)
    sims = all_vecs @ all_vecs.T
    pair_sims = [sims[i][j] for i, j in itertools.combinations(range(n), 2)]
    return {
        str(t): {
            "pairs_above_threshold": sum(1 for s in pair_sims if s >= t),
            "total_pairs": len(pair_sims),
        }
        for t in thresholds
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)