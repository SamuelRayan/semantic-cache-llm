"""Embedding-kNN router: for each candidate model, estimate P(success) from
similar past queries, shrunk toward that model's global success rate, then
apply a cheapest-good-enough decision rule.

Per "When Simple kNN Beats Complex Learned Routers" (arXiv 2505.12601) and
"The Routing Plateau" (arXiv 2606.07587): most of the routing value is in
predicting per-model correctness well, and a similarity-weighted kNN
estimator gets most of the way there without any training step — new
outcomes are usable as neighbors immediately, with no retrain. Static /
offline-trained routers (the handcrafted-features baseline in this repo)
plateau on instance-specific hard queries where the local neighborhood
signal kNN uses is exactly what's missing.
"""

import math
import random
import time
from dataclasses import dataclass, field

from app.routing.model_registry import model_registry
from app.routing.experience_store import experience_store
from app.routing.config import routing_settings


@dataclass
class RouteDecision:
    model: str
    p_hat: dict          # model_name -> estimated P(success)
    ess: dict             # model_name -> effective sample size behind that estimate
    reason: str           # threshold | utility | fallback_ood | explore | forced
    confidence: float     # p_hat of the chosen model
    routing_ms: float
    low_confidence: bool = False
    max_neighbor_sim: float = 0.0


def _weight(sim: float, tau: float) -> float:
    return math.exp((sim - 1.0) / tau)


class KnnRouter:
    def __init__(self, registry=None, store=None, settings=None):
        self.registry = registry or model_registry
        self.store = store or experience_store
        self.settings = settings or routing_settings

    def _estimate(self, neighbors, model_name: str):
        s = self.settings
        prior = self.store.global_success_rate(model_name)

        num = s.prior_strength * prior
        den = s.prior_strength
        sum_w = 0.0
        sum_w2 = 0.0

        for nb in neighbors:
            if nb["similarity"] < s.min_sim:
                continue
            outcome = nb["outcomes"].get(model_name)
            if outcome is None:
                continue
            w = _weight(nb["similarity"], s.tau)
            y = 1.0 if outcome["success"] else 0.0
            num += w * y
            den += w
            sum_w += w
            sum_w2 += w * w

        p_hat = num / den if den > 0 else prior
        ess = (sum_w * sum_w / sum_w2) if sum_w2 > 0 else 0.0
        return p_hat, ess

    def route(self, prompt: str, embedding, quality_target: float = None,
              mode: str = None) -> RouteDecision:
        start = time.time()
        s = self.settings
        quality_target = quality_target if quality_target is not None else s.quality_target
        mode = mode or s.decision_mode

        models = self.registry.enabled_models()
        neighbors = self.store.neighbors(embedding, k=s.k_neighbors, min_sim=0.0)
        max_sim = max((nb["similarity"] for nb in neighbors), default=0.0)

        p_hat, ess = {}, {}
        for m in models:
            p, e = self._estimate(neighbors, m.name)
            p_hat[m.name] = p
            ess[m.name] = e

        total_ess = sum(ess.values())
        low_confidence = (max_sim < s.ood_sim) or (total_ess < s.min_ess)

        if low_confidence:
            fallback_models = self.registry.by_tier(s.fallback_tier)
            chosen = fallback_models[0].name if fallback_models else models[0].name
            return RouteDecision(
                chosen, p_hat, ess, "fallback_ood",
                confidence=p_hat.get(chosen, 0.0),
                routing_ms=(time.time() - start) * 1000,
                low_confidence=True, max_neighbor_sim=max_sim,
            )

        if s.epsilon > 0 and random.random() < s.epsilon:
            for m in self.registry.cheapest_to_most_expensive():
                if (p_hat[m.name] >= quality_target - s.explore_margin
                        and ess[m.name] < s.explore_ess_threshold):
                    return RouteDecision(
                        m.name, p_hat, ess, "explore",
                        confidence=p_hat[m.name],
                        routing_ms=(time.time() - start) * 1000,
                        max_neighbor_sim=max_sim,
                    )

        if mode == "utility":
            lam = s.utility_lambda
            costs = {m.name: m.cost_per_1k_input + m.cost_per_1k_output for m in models}
            max_cost = max(costs.values()) or 1.0
            chosen = max(
                p_hat,
                key=lambda name: (1 - lam) * p_hat[name] - lam * (costs[name] / max_cost)
            )
            reason = "utility"
        else:
            chosen = None
            for m in self.registry.cheapest_to_most_expensive():
                if p_hat[m.name] >= quality_target:
                    chosen = m.name
                    break
            if chosen is None:
                chosen = max(p_hat, key=p_hat.get)
            reason = "threshold"

        return RouteDecision(
            chosen, p_hat, ess, reason,
            confidence=p_hat.get(chosen, 0.0),
            routing_ms=(time.time() - start) * 1000,
            max_neighbor_sim=max_sim,
        )


knn_router = KnnRouter()
