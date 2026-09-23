"""A mock provider for simulation/evaluation only: instead of actually
generating anything, it looks up the ground-truth pass/fail label for
(prompt, model) from the simulated dataset and fabricates a response whose
SURFACE signals (finish_reason, mean_logprob, occasional truncation) are
noisily correlated with that label — exactly the kind of imperfect signal a
real verifier has to work with. The router itself never sees the label
directly; it only ever sees these surface signals and, later, judged
quality scores, same as it would with a real LLM.
"""

import hashlib
import random


class MockTieredProvider:
    def __init__(self, lookup_fn, seed: int = 0):
        """lookup_fn(prompt, model) -> {"label": 0/1, "quality": float}"""
        self.lookup_fn = lookup_fn
        self.rng = random.Random(seed)

    def generate(self, prompt: str, system_prompt: str, model: str, temperature: float) -> dict:
        outcome = self.lookup_fn(prompt, model)
        label = outcome["label"]
        quality = outcome["quality"]

        seed = hashlib.md5((system_prompt + prompt + model).encode()).hexdigest()[:8]
        text = f"[{model}-{seed}] answer to: '{prompt[:60]}' (sim_quality={quality:.2f})"

        finish_reason = "stop"
        mean_logprob = -0.15 - 0.3 * self.rng.random()   # confident-looking by default

        if label == 0:
            r = self.rng.random()
            if r < 0.35:
                # Looks truncated — an easy, mostly-reliable failure signal
                finish_reason = "length"
                text = text[: max(1, len(text) // 6)]
            elif r < 0.60:
                # Looks uncertain but isn't obviously broken
                mean_logprob = -1.6 - 1.0 * self.rng.random()
            # else: confidently wrong — no surface signal at all. This is
            # exactly the class of failure a non-LLM verifier can't catch,
            # by design — it's what the judge/experience-store loop is for.

        input_tokens = max(1, len(prompt.split()))
        output_tokens = max(1, len(text.split()))

        return {
            "text": text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model": model,
            "finish_reason": finish_reason,
            "mean_logprob": mean_logprob,
            "latency_ms": 0.0,
        }
