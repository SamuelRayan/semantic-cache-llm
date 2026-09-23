from prometheus_client import Counter, Histogram, CollectorRegistry

registry = CollectorRegistry()

CACHE_HITS = Counter("semantic_cache_hits_total", "Total cache hits", registry=registry)
CACHE_MISSES = Counter("semantic_cache_misses_total", "Total cache misses", registry=registry)
COST_SAVED = Counter("semantic_cache_cost_saved_usd_total", "Estimated USD saved by cache hits", registry=registry)
REQUEST_LATENCY = Histogram(
    "semantic_cache_request_latency_seconds",
    "Latency per request",
    ["cache_status"],
    registry=registry,
)

# ── Routing metrics ──────────────────────────────────────────────────────────
ROUTED_REQUESTS = Counter(
    "routed_requests_total", "Requests served by each model/reason combination",
    ["model", "reason"], registry=registry,
)
ESCALATIONS = Counter(
    "escalations_total", "Cascade escalations from one tier to the next",
    ["from_model", "to_model"], registry=registry,
)
ROUTING_LATENCY = Histogram(
    "routing_latency_seconds", "Time spent making a routing decision (CPU only, no LLM call)",
    registry=registry,
)
COST_SAVED_VS_LARGE = Counter(
    "estimated_cost_saved_vs_large_usd_total",
    "Estimated USD saved by routing away from the large model",
    registry=registry,
)