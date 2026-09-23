"""Generates a deterministic synthetic dataset of ~3,000 prompts across 8
task families, with paraphrase clusters (for testing the semantic cache)
and per-model ground-truth correctness (for evaluating the router).

Usage (from the project root):
    python sim/generate_routing_dataset.py
    python sim/generate_routing_dataset.py --n-groups-per-family 150 --seed 7

Output: sim/data/routing_dataset.jsonl, one JSON object per prompt:
    {
      "query_id": "...", "prompt": "...", "group_id": "...",
      "family": "...", "difficulty": 1-5, "anomaly": bool,
      "labels": {"mock-small": {"label":0/1,"quality":float}, ...}
    }
"""

import argparse
import json
import math
import os
import random

FAMILIES = [
    "factual_lookup",            # 1
    "extraction_formatting",     # 2
    "classification",            # 3
    "summarization",             # 4
    "simple_code",                # 5
    "debugging_complex_code",     # 6
    "math_word_problems",         # 7
    "multistep_reasoning",        # 8
]

MODELS = ["mock-small", "mock-medium", "mock-large"]

# skill[model][family_index(1-8)] — higher means that model handles that
# family better. Chosen so: small is good at 1-3, medium adds 4-5, only
# large reliably handles 6-8. Sigmoid(skill - difficulty + noise) turns
# this into a success probability against the 1-5 difficulty scale.
SKILL = {
    "mock-small":  {1: 4.5, 2: 4.0, 3: 3.5, 4: 1.5, 5: 1.0, 6: 0.3, 7: 1.0, 8: -0.2},
    "mock-medium": {1: 5.0, 2: 4.8, 3: 4.5, 4: 4.0, 5: 3.5, 6: 2.0, 7: 2.5, 8: 1.3},
    "mock-large":  {1: 5.0, 2: 5.0, 3: 5.0, 4: 5.0, 5: 5.0, 6: 4.5, 7: 4.5, 8: 4.0},
}

# Difficulty distributions skew easier for the "easy" families and harder
# for the "hard" families, but every difficulty remains possible everywhere
# — this is what creates both "easy queries in a hard family" and "hard
# queries solvable only by large" within the SAME family.
DIFFICULTY_WEIGHTS = {
    "factual_lookup":            [30, 30, 20, 12, 8],
    "extraction_formatting":     [25, 30, 25, 12, 8],
    "classification":            [25, 28, 25, 14, 8],
    "summarization":             [18, 25, 27, 18, 12],
    "simple_code":                [22, 28, 25, 15, 10],
    "debugging_complex_code":     [8, 14, 24, 28, 26],
    "math_word_problems":         [15, 22, 26, 22, 15],
    "multistep_reasoning":        [6, 12, 22, 28, 32],
}

PARAPHRASE_WRAPPERS = [
    "{core}",
    "Quick question: {core}",
    "Could you help me with this? {core}",
    "I need to know this: {core}",
    "Here's what I'm working on — {core}",
]

FACT_TOPICS = [
    "is the capital of France", "is the boiling point of water at sea level in Celsius",
    "wrote 'Pride and Prejudice'", "is the largest planet in the solar system",
    "is the speed of light in a vacuum, in km/s", "is the chemical symbol for gold",
    "year did World War II end", "is the tallest mountain on Earth",
    "currency is used in Japan", "number of continents are there on Earth",
    "is the freezing point of water in Fahrenheit", "painted the Mona Lisa",
    "is the smallest prime number", "language has the most native speakers worldwide",
    "is the longest river in the world", "planet is known as the Red Planet",
    "year did the Berlin Wall fall", "is the capital of Australia",
    "element has atomic number 1", "invented the telephone",
]

SAMPLE_BLOBS = [
    "Contact us at support@example.com or sales@example.org before 2026-01-15. Price: $49.99.",
    "Order #4471 shipped on 03/14/2026 to john.doe@mail.com, total $120.50, call 555-0142.",
    "Meeting scheduled for 2026-02-01 with jane@corp.io. Budget: $15,000. Backup: 555-9981.",
    "Invoice due 2026-03-30, amount $899.00, reach billing@vendor.net or (555) 234-9090.",
]

SAMPLE_REVIEWS = [
    "This product completely exceeded my expectations, I would buy it again in a heartbeat.",
    "Absolutely terrible experience, it broke after two days and support never responded.",
    "It's fine, does what it says, nothing special but nothing wrong with it either.",
    "Best purchase I've made all year, the quality is outstanding and shipping was fast.",
    "Waste of money, the description was misleading and the material feels cheap.",
]

SAMPLE_PARAGRAPHS = [
    "The team spent the quarter migrating the legacy billing system to a new microservice "
    "architecture, which reduced average latency by 40% but introduced new operational "
    "overhead around service discovery and distributed tracing.",
    "Recent field trials showed that the new irrigation technique reduced water usage by a "
    "third while maintaining crop yield, though the upfront equipment cost remains a barrier "
    "for smaller farms in the region.",
    "The committee reviewed three proposals for the community center renovation and ultimately "
    "selected the one balancing cost, accessibility improvements, and minimal disruption to "
    "ongoing programs during construction.",
]

SIMPLE_CODE_TASKS = [
    "reverses a string", "checks if a number is prime", "returns the factorial of n",
    "finds the maximum value in a list", "counts vowels in a string",
    "checks if a string is a palindrome", "removes duplicates from a list",
    "converts Celsius to Fahrenheit", "sums the digits of an integer",
    "returns the nth Fibonacci number",
]

BUG_SCENARIOS = [
    "here's a stack trace ending in 'RecursionError: maximum recursion depth exceeded' — "
    "why is this happening and how do I fix the recursive function causing it?",
    "my multi-threaded worker pool occasionally deadlocks under load — help me find the race condition",
    "this async function silently swallows exceptions and I can't figure out where they're coming from",
    "the API returns 500 errors intermittently under high concurrency, but only in production",
    "my binary search implementation returns the wrong index for duplicate values in the array",
    "the memory usage of this long-running service grows unbounded over 24 hours — find the leak",
]

MATH_TEMPLATES = [
    "if a train travels at {a} mph for {c} hours, how far does it go?",
    "a rectangle has length {a} and width {b}, what is its area?",
    "if {a} workers can finish a job in {c} days, how many days would {b} workers take?",
    "a store discounts an item by {c}% from ${a}, what is the final price?",
    "what is the sum of the first {a} positive even numbers?",
]

REASONING_SCENARIOS = [
    "design a caching strategy for a read-heavy API with strict consistency requirements "
    "and a multi-region deployment — walk through the trade-offs",
    "plan a migration from a monolithic application to microservices, considering team "
    "structure, deployment risk, and data consistency during the transition",
    "compare using a message queue versus direct synchronous calls for an order-processing "
    "pipeline that must never lose an order, and recommend one with justification",
    "we need to reduce our cloud bill by 30% without hurting reliability — propose a "
    "step-by-step plan and explain the risks of each step",
    "design a rate-limiting scheme for a public API that's fair across tenants of very "
    "different traffic volumes, and justify the algorithm choice",
]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _core_for(family: str, rng: random.Random) -> str:
    if family == "factual_lookup":
        return f"what {rng.choice(FACT_TOPICS)}?"
    if family == "extraction_formatting":
        kind = rng.choice(["email addresses", "phone numbers", "dates", "dollar amounts"])
        blob = rng.choice(SAMPLE_BLOBS)
        return f'extract all {kind} from this text: "{blob}"'
    if family == "classification":
        text = rng.choice(SAMPLE_REVIEWS)
        return f'classify the sentiment of this review as positive, negative, or neutral: "{text}"'
    if family == "summarization":
        text = rng.choice(SAMPLE_PARAGRAPHS)
        return f'summarize this paragraph in one sentence: "{text}"'
    if family == "simple_code":
        task = rng.choice(SIMPLE_CODE_TASKS)
        return f"write a python function that {task}"
    if family == "debugging_complex_code":
        return rng.choice(BUG_SCENARIOS)
    if family == "math_word_problems":
        a, b, c = rng.randint(2, 60), rng.randint(2, 40), rng.randint(2, 24)
        template = rng.choice(MATH_TEMPLATES)
        return template.format(a=a, b=b, c=c)
    if family == "multistep_reasoning":
        return rng.choice(REASONING_SCENARIOS)
    raise ValueError(family)


def _sample_quality(label: int, rng: random.Random) -> float:
    base = 1.0 if label else 0.0
    noise = rng.gauss(0, 0.08)
    return max(0.0, min(1.0, base + noise))


def generate_dataset(n_groups_per_family: int = 125, seed: int = 42):
    rng = random.Random(seed)
    rows = []

    for family in FAMILIES:
        family_idx = FAMILIES.index(family) + 1
        weights = DIFFICULTY_WEIGHTS[family]

        for g in range(n_groups_per_family):
            group_id = f"{family}_{g}"
            difficulty = rng.choices([1, 2, 3, 4, 5], weights=weights)[0]
            # ~5% of groups are deliberately non-monotone: medium unusually
            # bad, small unusually good — keeps the problem from being
            # perfectly ordered by tier, per the spec.
            anomaly = rng.random() < 0.05

            core = _core_for(family, rng)
            n_paraphrases = rng.randint(2, 4)
            wrappers = rng.sample(PARAPHRASE_WRAPPERS, k=min(n_paraphrases, len(PARAPHRASE_WRAPPERS)))

            # Ground truth is computed ONCE per (group, model) — paraphrasing
            # a question doesn't change whether a model can answer it.
            labels = {}
            for model in MODELS:
                skill = SKILL[model][family_idx]
                if anomaly and model == "mock-medium":
                    skill -= 3.0
                if anomaly and model == "mock-small":
                    skill += 1.5
                noise = rng.gauss(0, 0.6)
                p_success = sigmoid(skill - difficulty + noise)
                label = 1 if rng.random() < p_success else 0
                labels[model] = {
                    "label": label,
                    "quality": round(_sample_quality(label, rng), 4),
                    "p_success": round(p_success, 4),
                }

            for i, wrapper in enumerate(wrappers):
                prompt = wrapper.format(core=core)
                rows.append({
                    "query_id": f"{group_id}_p{i}",
                    "prompt": prompt,
                    "group_id": group_id,
                    "family": family,
                    "difficulty": difficulty,
                    "anomaly": anomaly,
                    "labels": labels,
                })

    rng.shuffle(rows)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-groups-per-family", type=int, default=125)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "data", "routing_dataset.jsonl"))
    args = ap.parse_args()

    rows = generate_dataset(args.n_groups_per_family, args.seed)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    families = {}
    for r in rows:
        families[r["family"]] = families.get(r["family"], 0) + 1

    print(f"Wrote {len(rows)} prompts across {len(set(r['group_id'] for r in rows))} groups to {args.out}")
    print("Per-family counts:")
    for fam, n in families.items():
        print(f"  {fam}: {n}")
    anomalies = sum(1 for r in rows if r["anomaly"])
    print(f"Anomalous (non-monotone) groups: {sum(1 for g in set(r['group_id'] for r in rows if r['anomaly']))}")
    print(f"Rows belonging to an anomalous group: {anomalies}")


if __name__ == "__main__":
    main()
