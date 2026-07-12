import time
import itertools
from fastapi import FastAPI, Response
from pydantic import BaseModel
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.cache_engine import get_or_generate, _system_prompt_hash
from app.cache_db import cache_db
from app.vector_store import vector_store
from app.metrics import CACHE_HITS, CACHE_MISSES, REQUEST_LATENCY, COST_SAVED, registry

app = FastAPI(title="Semantic Cache Proxy")

class ChatRequest(BaseModel):
    prompt: str
    system_prompt: str = "You are a helpful assistant."
    model: str = "gpt-4o-mini"
    temperature: float = 0.7

class InvalidateRequest(BaseModel):
    system_prompt: str
    model: str = "gpt-4o-mini"
    temperature: float = 0.7

@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest, response: Response):
    start = time.time()
    result = get_or_generate(req.prompt, req.system_prompt, req.model, req.temperature)
    duration = time.time() - start

    response.headers["X-Cache-Status"] = result.cache_status
    if result.similarity is not None:
        response.headers["X-Cache-Similarity"] = f"{result.similarity:.4f}"

    if result.cache_status == "HIT":
        CACHE_HITS.inc()
        COST_SAVED.inc(0.002)  # what we would have spent had this been a miss, roughly
    else:
        CACHE_MISSES.inc()

    REQUEST_LATENCY.labels(cache_status=result.cache_status).observe(duration)

    return {
        "id": "chatcmpl-mock",
        "model": req.model,
        "cache_status": result.cache_status,
        "cache_similarity": result.similarity,
        "choices": [{"message": {"role": "assistant", "content": result.text}}],
    }

@app.post("/v1/cache/invalidate")
def invalidate(req: InvalidateRequest):
    bucket = _system_prompt_hash(req.system_prompt, req.model, req.temperature)
    cache_db.delete_by_system_prompt(bucket)
    return {"status": "invalidated", "bucket": bucket}

@app.get("/v1/cache/stats")
def stats():
    return {"total_entries": cache_db.count()}

@app.get("/v1/cache/threshold-report")
def threshold_report():
    """Simplified but real threshold tradeoff tool: pairwise-compares every
    cached embedding against every other and shows, at each candidate
    threshold, how many pairs would be considered 'the same question'.
    Fine at demo scale (hundreds of entries); at production scale you'd log
    near-misses in production traffic and replay those instead of doing a
    full pairwise comparison."""
    thresholds = [0.85, 0.90, 0.92, 0.95, 0.98]
    n = vector_store.index.ntotal
    if n < 2:
        return {"message": "Not enough cached entries yet. Send a few requests first."}
    all_vectors = vector_store.index.reconstruct_n(0, n)
    sims = all_vectors @ all_vectors.T
    pair_sims = [sims[i][j] for i, j in itertools.combinations(range(n), 2)]
    report = {}
    for t in thresholds:
        would_hit = sum(1 for s in pair_sims if s >= t)
        report[str(t)] = {"pairs_above_threshold": would_hit, "total_pairs": len(pair_sims)}
    return report

@app.get("/metrics")
def metrics():
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)