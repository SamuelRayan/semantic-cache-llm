# Semantic Caching Layer for LLM APIs

A drop-in caching proxy that sits in front of LLM API calls, detects when two
prompts mean the same thing even if worded differently, and serves the cached
answer instantly instead of paying for and waiting on another model call.

## The problem

Most LLM-powered applications get asked the same underlying questions over and
over, phrased differently by different users. Exact-string caching misses all
of this — "What is the capital of France?" and "Can you tell me France's
capital city?" are different strings but the same question, and a naive cache
treats them as two separate, billable API calls. At any real scale, that's
redundant spend and redundant latency on every single repeat.

## Results

Load-tested with 300 requests built from 8 topics rephrased 5 different ways
each (simulating how real users actually ask the same question differently):

| Metric | Result |
|---|---|
| Cache hit rate | **95.0%** |
| Avg latency — cache hit | **85.6ms** |
| Avg latency — cache miss | **768.7ms** |
| Latency improvement on hits | **~9x faster** |

![Grafana dashboard showing hit rate, cumulative cost saved, and P95 latency by cache status](docs/dashboard.png)

## How it works

1. Every incoming prompt is embedded locally using `sentence-transformers`
   (`all-MiniLM-L6-v2`) — no external API call, no cost, runs entirely on CPU.
2. The embedding is compared against previously cached prompts using FAISS
   (`IndexFlatIP`, cosine similarity via normalized inner product).
3. If a sufficiently similar prompt exists **and** it shares the same system
   prompt, model, and temperature (its "bucket") **and** hasn't expired, the
   cached response is returned — no LLM call made.
4. Otherwise, the request goes to the LLM, and both the response and its
   embedding are stored for future lookups.
5. Cache entries are assigned a TTL automatically — short for time-sensitive
   queries (detected by keywords like "today," "current," "latest"), long for
   stable factual queries.

## Architecture decisions

**FAISS + SQLite instead of Redis + RedisVL.** The reference architecture for
this kind of system typically uses Redis. I chose an in-process FAISS index
with SQLite for metadata instead, since this is a single-instance deployment
and doesn't yet need shared state across multiple API replicas. The interfaces
are cleanly separated, so moving to Redis/RedisVL or Qdrant for horizontal
scaling later is a contained change, not a rewrite.

**A mock LLM provider behind a real interface.** The caching logic — embed,
search, compare, expire, store — doesn't depend on what the LLM actually
says. So the entire system was built and load-tested against a deterministic
mock provider, at zero cost and with no API key required, behind an abstract
`LLMProvider` interface. A working `OpenAIProvider` implementation exists and
is one config change away from being live (`USE_MOCK_LLM=false` +
`OPENAI_API_KEY`) — nothing else in the codebase needs to change.

**Bucketing by system prompt + model + temperature.** Two identical user
prompts sent with different system prompts, models, or temperatures must
never share a cache entry, since they're not actually asking for the same
thing. Each of those three values is hashed together into a "bucket," and
matches are filtered by bucket before being considered valid hits.

## Tech stack

Python 3.11, FastAPI, sentence-transformers, FAISS, SQLite, Prometheus,
Grafana, Docker Compose.

## Running it

```bash
docker compose up -d
```

This starts three services:
- `cache-api` — the FastAPI proxy on `localhost:8000`
- `prometheus` — metrics scraping on `localhost:9090`
- `grafana` — dashboards on `localhost:3000` (login: admin/admin)

Send a request:
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of France?"}'
```

The response includes an `X-Cache-Status: HIT` or `MISS` header, and
`X-Cache-Similarity` on hits.

Run the load test:
```bash
python load_test.py
```

## Project structure

```
semantic-cache/
├── app/
│   ├── config.py          # all tunable settings (similarity threshold, TTLs, etc.)
│   ├── embeddings.py       # local embedding model wrapper
│   ├── vector_store.py     # FAISS index wrapper
│   ├── cache_db.py         # SQLite metadata/response store
│   ├── llm_providers.py    # LLMProvider interface + Mock/OpenAI implementations
│   ├── ttl_policy.py       # auto-assigns cache expiry per prompt
│   ├── cache_engine.py     # core get-or-generate logic
│   ├── metrics.py          # Prometheus instrumentation
│   └── main.py             # FastAPI app
├── docker/prometheus.yml
├── Dockerfile
├── docker-compose.yml
├── load_test.py
└── requirements.txt
```

## What I'd do next with more time

- Move to Qdrant or RedisVL for multi-instance shared state
- Adaptive per-task-type similarity thresholds instead of one global value
- Near-miss logging in production traffic to tune the threshold with real data instead of synthetic pairwise comparison
- Streaming response support for cache misses