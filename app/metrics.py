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