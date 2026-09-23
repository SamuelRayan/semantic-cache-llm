import json
import threading
from app.cache_db import cache_db
from app.llm_providers import get_provider

# Sensitivity score thresholds — used by the three-layer lookup
# to decide how aggressively a cached entry can be reused.
THRESHOLD_FREE   = 0.3   # score < 0.3  → safe to serve to anyone
THRESHOLD_MEDIUM = 0.7   # score < 0.7  → serve only to similar contexts
                          # score >= 0.7 → almost never reuse

SCORER_PROMPT = """You are evaluating whether an answer depends on prior conversation context.

Question asked: {prompt}

Prior context summary (what was discussed before this question):
{context_summary}

Answer given:
{answer}

Score from 0.0 to 1.0 — how much does this answer depend on the specific prior context?

0.0 = this answer would be word-for-word identical even with zero prior context (a definition, a formula, a factual lookup)
0.5 = this answer uses some context but the core information is universally valid
1.0 = this answer only makes sense given this specific prior context; serving it to someone else would be wrong

Return ONLY a JSON object, nothing else, no markdown:
{{"score": 0.0, "reason": "one short sentence"}}"""


def _compute_score(prompt: str, context_summary: str, answer: str) -> float:
    """Shared scoring logic used by both the async and synchronous paths.
    Calls the LLM once and parses its JSON verdict, falling back to a
    neutral 0.5 if the provider doesn't return valid JSON (this is always
    true for MockProvider, which returns a fixed fake string — see
    llm_providers.py)."""
    filled = SCORER_PROMPT.format(
        prompt=prompt,
        context_summary=context_summary if context_summary else "(none — cold start)",
        answer=answer[:500],   # cap to avoid huge token bills on long answers
    )
    provider = get_provider()
    result = provider.generate(
        prompt=filled,
        system_prompt="You are a precise JSON-only evaluator.",
        model="gpt-4o-mini",
        temperature=0.0,
    )

    text = result["text"].strip()

    # MockProvider returns fake text, not JSON — handle gracefully
    # by assigning a neutral mid-range score so mock runs still work
    try:
        parsed = json.loads(text)
        score = float(parsed.get("score", 0.5))
        score = max(0.0, min(1.0, score))   # clamp to [0,1]
    except (json.JSONDecodeError, ValueError):
        # If the mock (or a misbehaving real provider) doesn't return
        # valid JSON, default to 0.5 — neutral, not wrong
        score = 0.5

    return score


def score_synchronous(prompt: str, context_summary: str, answer: str) -> float:
    """Runs scoring inline, blocking, and returns the float score directly.
    Used when SYNC_SCORING=true (tests, low-latency environments) so the
    sensitivity score is written to the DB before the caller returns —
    otherwise Layer 2 can never fire on an immediately-following request,
    since score_in_background() writes the score after the response has
    already gone out."""
    try:
        return _compute_score(prompt, context_summary, answer)
    except Exception as e:
        print(f"[scorer] synchronous scoring failed: {e}")
        return 0.5


def _score_async(cache_id: str, prompt: str,
                 context_summary: str, answer: str):
    """Runs in a background thread after the response is already
    returned to the user — zero added latency."""
    try:
        score = _compute_score(prompt, context_summary, answer)
        cache_db.update_sensitivity_score(cache_id, score)
    except Exception as e:
        # Scoring is best-effort. A failure here must never affect the
        # user-facing response — log and move on.
        print(f"[scorer] background scoring failed for {cache_id}: {e}")


def score_in_background(cache_id: str, prompt: str,
                        context_summary: str, answer: str):
    """Fire-and-forget. Returns immediately; scoring happens in a
    daemon thread that dies if the main process exits."""
    t = threading.Thread(
        target=_score_async,
        args=(cache_id, prompt, context_summary, answer),
        daemon=True,
    )
    t.start()
