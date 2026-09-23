"""The set of models the router can choose between, with their relative
pricing. Loaded from config/models.yaml so the roster can change without
touching code."""

import os
import yaml
from dataclasses import dataclass


@dataclass
class ModelSpec:
    name: str
    tier: str                  # "small" | "medium" | "large"
    cost_per_1k_input: float
    cost_per_1k_output: float
    avg_latency_ms: float
    enabled: bool = True


_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "models.yaml"
)


class ModelRegistry:
    def __init__(self, config_path: str = None):
        config_path = config_path or _DEFAULT_CONFIG_PATH
        with open(config_path) as f:
            data = yaml.safe_load(f)

        self.models = {m["name"]: ModelSpec(**m) for m in data["models"]}

        self._by_tier = {}
        for m in self.models.values():
            self._by_tier.setdefault(m.tier, []).append(m)
        for tier_models in self._by_tier.values():
            tier_models.sort(key=lambda m: m.cost_per_1k_input + m.cost_per_1k_output)

    def get(self, name: str) -> ModelSpec | None:
        return self.models.get(name)

    def enabled_models(self) -> list[ModelSpec]:
        return [m for m in self.models.values() if m.enabled]

    def by_tier(self, tier: str) -> list[ModelSpec]:
        return [m for m in self._by_tier.get(tier, []) if m.enabled]

    def cheapest_to_most_expensive(self) -> list[ModelSpec]:
        return sorted(self.enabled_models(),
                      key=lambda m: m.cost_per_1k_input + m.cost_per_1k_output)

    def cost(self, name: str, input_tokens: int, output_tokens: int) -> float:
        m = self.models[name]
        return (
            (input_tokens / 1000) * m.cost_per_1k_input +
            (output_tokens / 1000) * m.cost_per_1k_output
        )

    def tier_of(self, name: str) -> str | None:
        m = self.models.get(name)
        return m.tier if m else None


model_registry = ModelRegistry()
