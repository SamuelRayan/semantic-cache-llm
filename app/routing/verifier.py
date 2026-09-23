"""Non-LLM verifier: cheap, deterministic checks on a response, used to
catch an obviously-bad answer from a cheap model and escalate once —
FrugalGPT-style cascading, but the escalation trigger is a free heuristic
check rather than a second model call."""

import json
from dataclasses import dataclass

from app.routing.model_registry import model_registry as _default_registry

REFUSAL_PATTERNS = [
    "i cannot help with that", "i can't assist", "i'm not able to",
    "as an ai", "i cannot provide", "sorry, i can't", "i am unable to",
]

_TIER_ORDER = ["small", "medium", "large"]


@dataclass
class VerifyResult:
    passed: bool
    reason: str = "ok"


def verify_response(response_text: str, finish_reason: str = None,
                     mean_logprob: float = None, expects_json: bool = False,
                     min_chars: int = 5, min_logprob: float = -1.5) -> VerifyResult:
    text = (response_text or "").strip()

    if len(text) < min_chars:
        return VerifyResult(False, "empty_or_too_short")

    lowered = text.lower()
    if any(p in lowered for p in REFUSAL_PATTERNS):
        return VerifyResult(False, "refusal")

    if finish_reason == "length":
        return VerifyResult(False, "truncated")

    if expects_json:
        try:
            json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return VerifyResult(False, "invalid_json")

    if mean_logprob is not None and mean_logprob < min_logprob:
        return VerifyResult(False, "low_confidence_logprob")

    return VerifyResult(True, "ok")


def next_tier(current_tier: str) -> str | None:
    if current_tier not in _TIER_ORDER:
        return None
    idx = _TIER_ORDER.index(current_tier)
    if idx + 1 >= len(_TIER_ORDER):
        return None
    return _TIER_ORDER[idx + 1]


def cascade_generate(provider, prompt: str, system_prompt: str, temperature: float,
                      chosen_model_spec, registry=None, store=None,
                      query_id: str = None, embedding=None,
                      expects_json: bool = False, settings=None):
    """Generate with the routed model; on verifier failure, escalate exactly
    one tier and retry once (at most one escalation, per spec).

    Returns (result_dict, model_used, escalated: bool, verify_reason).
    """
    from app.routing.config import routing_settings as _rs
    settings = settings or _rs
    registry = registry or _default_registry

    result = provider.generate(prompt, system_prompt, chosen_model_spec.name, temperature)
    v = verify_response(
        result.get("text", ""),
        finish_reason=result.get("finish_reason"),
        mean_logprob=result.get("mean_logprob"),
        expects_json=expects_json,
        min_chars=settings.verifier_min_response_chars,
        min_logprob=settings.verifier_min_logprob,
    )

    if v.passed:
        return result, chosen_model_spec.name, False, v.reason

    if store is not None and query_id is not None and embedding is not None:
        # Log the failure as a quality-0 outcome so the router learns this
        # model doesn't work here, without waiting on the (sampled) judge.
        store.add_outcome(query_id, embedding, prompt, chosen_model_spec.name,
                           quality=0.0, cost=0.0, latency=0.0, source="verifier")

    nxt_tier = next_tier(chosen_model_spec.tier)
    if nxt_tier is None:
        return result, chosen_model_spec.name, False, v.reason  # already at the top tier

    candidates = registry.by_tier(nxt_tier)
    if not candidates:
        return result, chosen_model_spec.name, False, v.reason

    escalated_spec = candidates[0]
    result2 = provider.generate(prompt, system_prompt, escalated_spec.name, temperature)
    return result2, escalated_spec.name, True, v.reason
