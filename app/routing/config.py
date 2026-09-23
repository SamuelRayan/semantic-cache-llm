"""Central hyperparameters for the routing subsystem. Same pattern as
app/config.py — env-var overridable, one shared singleton."""

import os
from dataclasses import dataclass


def _f(name, default):
    return float(os.getenv(name, default))


def _i(name, default):
    return int(os.getenv(name, default))


def _b(name, default):
    return os.getenv(name, str(default)).lower() == "true"


@dataclass
class RoutingSettings:
    # Quality-label semantics
    pass_mark: float = _f("ROUTING_PASS_MARK", 0.7)          # quality >= this => "success"

    # kNN estimation
    min_sim: float = _f("ROUTING_MIN_SIM", 0.55)              # neighbor sim floor to count at all
    tau: float = _f("ROUTING_TAU", 0.1)                       # similarity-weight decay
    prior_strength: float = _f("ROUTING_PRIOR_STRENGTH", 3.0)  # alpha, shrinkage toward global prior
    k_neighbors: int = _i("ROUTING_K_NEIGHBORS", 30)           # FAISS top-k fetched per routing call

    # Decision rule
    decision_mode: str = os.getenv("ROUTING_DECISION_MODE", "threshold")  # "threshold" | "utility"
    quality_target: float = _f("ROUTING_QUALITY_TARGET", 0.8)
    utility_lambda: float = _f("ROUTING_UTILITY_LAMBDA", 0.3)  # cost weight in utility mode

    # Out-of-distribution fallback
    ood_sim: float = _f("ROUTING_OOD_SIM", 0.45)               # below this max-sim => low confidence
    min_ess: float = _f("ROUTING_MIN_ESS", 2.0)                # below this total ESS => low confidence
    fallback_tier: str = os.getenv("ROUTING_FALLBACK_TIER", "medium")

    # Exploration (off by default in production)
    epsilon: float = _f("ROUTING_EPSILON", 0.0)
    explore_margin: float = _f("ROUTING_EXPLORE_MARGIN", 0.05)
    explore_ess_threshold: float = _f("ROUTING_EXPLORE_ESS_THRESHOLD", 5.0)

    # Judge / shadow sampling
    judge_sample_rate: float = _f("ROUTING_JUDGE_SAMPLE_RATE", 0.2)
    shadow_sample_rate: float = _f("ROUTING_SHADOW_SAMPLE_RATE", 0.0)

    # Verifier
    verifier_min_response_chars: int = _i("ROUTING_MIN_RESPONSE_CHARS", 5)
    verifier_min_logprob: float = _f("ROUTING_MIN_LOGPROB", -1.5)


routing_settings = RoutingSettings()
