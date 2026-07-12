import hashlib
import time
import uuid
from app.embeddings import embedding_service
from app.vector_store import vector_store
from app.cache_db import cache_db
from app.llm_providers import get_provider
from app.ttl_policy import assign_ttl
from app.config import settings

def _system_prompt_hash(system_prompt: str,model : str, temperature:float) -> str:
    raw=f"{system_prompt} | {model} | {temperature}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

class CacheResult:
    def __init__(self,text,cache_status, similarity=None,latency_ms=None, cost_usd=0.0):
        self.text=text
        self.cache_status=cache_status
        self.similarity=similarity
        self.latency_ms=latency_ms
        self.cost_usd=cost_usd
        
MOCK_PRICE_PER_1K_INPUT=0.005
MOCK_PRICE_PER_1K_OUTPUT=0.015

def _estimate_cost(input_tokens, output_tokens):
    return (input_tokens/1000) * MOCK_PRICE_PER_1K_INPUT +(output_tokens/1000) * MOCK_PRICE_PER_1K_OUTPUT

def get_or_generate(prompt: str, system_prompt: str = "You are a helpful assistant.", model: str = "gpt-4o-mini", temperature: float = 0.7) -> CacheResult:
    start = time.time()
    bucket = _system_prompt_hash(system_prompt, model, temperature)
    query_vec = embedding_service.embed(prompt)

    candidates = vector_store.search(query_vec, top_k=5)
    now = time.time()

    for cache_id, score in candidates:
        entry = cache_db.get(cache_id)
        if entry is None:
            continue
        if entry["system_prompt_hash"] != bucket:
            continue
        if entry["expires_at"] < now:
            continue
        if score >= settings.similarity_threshold:
            cache_db.register_hit(cache_id)
            latency_ms = (time.time() - start) * 1000
            return CacheResult(entry["response_text"], "HIT", similarity=score, latency_ms=latency_ms, cost_usd=0.0)

    # loop finished with no acceptable match -> this is a MISS
    provider = get_provider()
    result = provider.generate(prompt, system_prompt, model, temperature)
    cost = _estimate_cost(result["input_tokens"], result["output_tokens"])

    cache_id = str(uuid.uuid4())
    ttl = assign_ttl(prompt)
    cache_db.insert({
        "id": cache_id,
        "prompt_text": prompt,
        "response_text": result["text"],
        "model": model,
        "system_prompt_hash": bucket,
        "temperature": temperature,
        "created_at": now,
        "expires_at": now + ttl,
    })
    vector_store.add(cache_id, query_vec)

    latency_ms = (time.time() - start) * 1000
    return CacheResult(result["text"], "MISS", similarity=None, latency_ms=latency_ms, cost_usd=cost)