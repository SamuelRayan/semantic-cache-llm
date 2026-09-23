"""Evaluates the kNN router against baselines (handcrafted-features HGB,
embedding-only logistic regression, and simple non-learned baselines),
plus an online-learning experiment and a full-system (cache + router)
replay. Everything runs on the synthetic dataset from
sim/generate_routing_dataset.py — no API keys, no network calls.

Usage (from the project root):
    python eval/run_routing_eval.py
    python eval/run_routing_eval.py --dataset routerbench   # best-effort, optional

Outputs (eval/results/):
    summary.csv            cost-quality sweep, one row per (router, quality_target)
    cost_quality_curve.png
    online_learning.png
    full_system.json
"""

import argparse
import csv
import json
import os
import random
import shutil
import sys
import time
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "sim", "data", "routing_dataset.jsonl")
MODEL_NAMES = ["mock-small", "mock-medium", "mock-large"]

QT_SWEEP = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
RANDOM_WEIGHT_SWEEP = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# ── data loading / splitting ─────────────────────────────────────────────────

def load_dataset(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def split_by_group(rows, test_frac=0.2, seed=42):
    """Split by group_id, never by row — paraphrases of the same underlying
    query must never appear on both sides, or the split leaks."""
    groups = sorted(set(r["group_id"] for r in rows))
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_test = max(1, int(len(groups) * test_frac))
    test_groups = set(groups[:n_test])
    train = [r for r in rows if r["group_id"] not in test_groups]
    test = [r for r in rows if r["group_id"] in test_groups]
    return train, test


# ── cost model (shared, mock pricing from the model registry) ──────────────

def est_tokens(prompt):
    input_tokens = max(1, len(prompt.split()))
    output_tokens = max(20, input_tokens // 2)
    return input_tokens, output_tokens


def make_cost_fn(model_registry):
    def cost_of(model_name, prompt):
        it, ot = est_tokens(prompt)
        return model_registry.cost(model_name, it, ot)
    return cost_of


# ── router adapters: everything reduces to route(prompt, emb, qt) -> (model, info) ─

def fixed_route_fn(model_name):
    def route(prompt, emb, qt):
        return model_name, {"reason": "fixed", "routing_ms": 0.0}
    return route


def oracle_route_fn(cheap_to_expensive):
    def route(prompt, emb, qt, row=None):
        for m in cheap_to_expensive:
            if row["labels"][m]["label"] == 1:
                return m, {"reason": "oracle", "routing_ms": 0.0}
        return cheap_to_expensive[-1], {"reason": "oracle_none_succeed", "routing_ms": 0.0}
    return route


def random_route_fn(large_weight, rng):
    def route(prompt, emb, qt):
        t0 = time.perf_counter()
        r = rng.random()
        if r < large_weight:
            chosen = "mock-large"
        elif r < large_weight + (1 - large_weight) / 2:
            chosen = "mock-medium"
        else:
            chosen = "mock-small"
        return chosen, {"reason": "random", "routing_ms": (time.perf_counter() - t0) * 1000}
    return route


def knn_route_fn(knn_router_obj):
    def route(prompt, emb, qt):
        d = knn_router_obj.route(prompt, emb, quality_target=qt)
        return d.model, {"reason": d.reason, "routing_ms": d.routing_ms}
    return route


def baseline_route_fn(baseline_obj):
    def route(prompt, emb, qt):
        d = baseline_obj.route(prompt, emb, quality_target=qt)
        return d.model, {"reason": d.reason, "routing_ms": d.routing_ms}
    return route


# ── evaluation loop ──────────────────────────────────────────────────────────

def evaluate_router(route_fn, test_rows, test_embs, qt, cost_fn, registry,
                     cascade=False, oracle=False, cascade_rng=None,
                     next_tier_fn=None):
    n = len(test_rows)
    total_quality = 0.0
    total_cost = 0.0
    large_count = 0
    severe_errors = 0
    escalations = 0
    latencies = []
    cascade_rng = cascade_rng or random.Random(0)

    for row, emb in zip(test_rows, test_embs):
        if oracle:
            chosen, info = route_fn(row["prompt"], emb, qt, row=row)
        else:
            chosen, info = route_fn(row["prompt"], emb, qt)
        latencies.append(info.get("routing_ms", 0.0))

        orig_chosen = chosen
        label = row["labels"][chosen]["label"]
        quality = row["labels"][chosen]["quality"]
        cost = cost_fn(chosen, row["prompt"])

        if cascade and label == 0:
            # Same noisy catch-rate as MockTieredProvider: ~60% of failures
            # show a surface signal (truncation/low-logprob) the non-LLM
            # verifier can actually catch; the rest are confidently wrong
            # and invisible to it.
            caught = cascade_rng.random() < 0.6
            if caught:
                tier = registry.tier_of(chosen)
                nxt = next_tier_fn(tier)
                if nxt:
                    candidates = registry.by_tier(nxt)
                    if candidates:
                        esc_model = candidates[0].name
                        escalations += 1
                        chosen = esc_model
                        label = row["labels"][esc_model]["label"]
                        quality = row["labels"][esc_model]["quality"]
                        cost += cost_fn(esc_model, row["prompt"])

        total_quality += quality
        total_cost += cost
        if chosen == "mock-large":
            large_count += 1

        only_large_succeeds = (
            row["labels"]["mock-large"]["label"] == 1
            and row["labels"]["mock-medium"]["label"] == 0
            and row["labels"]["mock-small"]["label"] == 0
        )
        if only_large_succeeds and orig_chosen == "mock-small":
            severe_errors += 1

    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else 0.0
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0.0

    return {
        "avg_quality": total_quality / n,
        "avg_cost": total_cost / n,
        "pct_large": large_count / n,
        "severe_error_rate": severe_errors / n,
        "escalation_rate": escalations / n,
        "routing_p50_ms": p50,
        "routing_p95_ms": p95,
        "n": n,
    }


def cost_to_reach_95pct(points, baseline_quality):
    """points: list of (cost, quality) for one router's sweep. Returns the
    lowest cost at which quality >= 95% of the always-large baseline, or
    None if the router never gets there."""
    target = 0.95 * baseline_quality
    qualifying = [c for c, q in points if q >= target]
    return min(qualifying) if qualifying else None


def normalized_auc(points, global_min_cost, global_max_cost, n_samples=200):
    """Trapezoidal AUC over (cost, quality), but resampled onto the SAME
    shared [global_min_cost, global_max_cost] grid for every router (flat
    extrapolation beyond each router's own observed sweep), then normalized
    by the total x-range. Comparing raw AUC over each router's own cost
    range is misleading — a router whose sweep happens to span a wider
    slice of the cost axis (e.g. random, which ranges from near-zero to
    near-always-large cost) accumulates more area purely from integrating
    over more width, even while sitting strictly below better routers on
    quality at every comparable cost. Resampling onto one shared domain
    fixes that."""
    if len(points) < 2 or global_max_cost <= global_min_cost:
        return None
    pts = sorted(points, key=lambda p: p[0])
    xs = np.array([c for c, _ in pts], dtype="float64")
    ys = np.array([q for _, q in pts], dtype="float64")
    grid = np.linspace(global_min_cost, global_max_cost, n_samples)
    interp_y = np.interp(grid, xs, ys, left=ys[0], right=ys[-1])
    return float(np.trapz(interp_y, grid) / (global_max_cost - global_min_cost))


# ── online-learning experiment ──────────────────────────────────────────────

def rolling(vals, window=50):
    out = []
    dq = deque()
    s = 0.0
    for v in vals:
        dq.append(v)
        s += v
        if len(dq) > window:
            s -= dq.popleft()
        out.append(s / len(dq))
    return out


def run_online_learning_experiment(train_rows, train_embs, test_rows, test_embs,
                                    seed, model_registry, cost_fn):
    from app.routing.experience_store import ExperienceStore
    from app.routing.knn_router import KnnRouter
    from app.routing.baseline_router import HandcraftedRouter
    from app.routing.config import RoutingSettings

    rng = random.Random(seed + 1)
    train_groups = sorted(set(r["group_id"] for r in train_rows))
    rng.shuffle(train_groups)
    n_boot = max(1, int(len(train_groups) * 0.10))
    boot_groups = set(train_groups[:n_boot])
    boot_idx = [i for i, r in enumerate(train_rows) if r["group_id"] in boot_groups]
    boot_rows = [train_rows[i] for i in boot_idx]
    boot_embs = train_embs[boot_idx]
    print(f"  online-learning bootstrap: {len(boot_groups)} groups / {len(boot_rows)} rows (10% of train)")

    online_dir = os.path.join(os.path.dirname(__file__), "..", "data", "eval_online_knn")
    shutil.rmtree(online_dir, ignore_errors=True)
    online_store = ExperienceStore(data_dir=online_dir, pass_mark=0.7)

    def bootstrap_rows():
        for row, emb in zip(boot_rows, boot_embs):
            for m in MODEL_NAMES:
                yield {
                    "query_id": row["query_id"], "embedding": emb, "prompt": row["prompt"],
                    "model": m, "quality": row["labels"][m]["quality"],
                    "cost": cost_fn(m, row["prompt"]), "latency": 100, "source": "offline",
                }
    online_store.bootstrap_from_dataset(bootstrap_rows())
    online_knn = KnnRouter(registry=model_registry, store=online_store, settings=RoutingSettings())

    labels_per_model = {m: [r["labels"][m]["label"] for r in boot_rows] for m in MODEL_NAMES}
    hgb_once = HandcraftedRouter(registry=model_registry, quality_target=0.8)
    hgb_once.fit([r["prompt"] for r in boot_rows], list(boot_embs), labels_per_model)

    knn_q, knn_c, hgb_q, hgb_c = [], [], [], []
    for row, emb in zip(test_rows, test_embs):
        d = online_knn.route(row["prompt"], emb, quality_target=0.8)
        q = row["labels"][d.model]["quality"]
        c = cost_fn(d.model, row["prompt"])
        knn_q.append(q); knn_c.append(c)
        if rng.random() < 0.2:   # judge sampling rate
            online_store.add_outcome(row["query_id"], emb, row["prompt"], d.model, q, c, 100, "online_judge")

        d2 = hgb_once.route(row["prompt"], emb, quality_target=0.8)
        q2 = row["labels"][d2.model]["quality"]
        c2 = cost_fn(d2.model, row["prompt"])
        hgb_q.append(q2); hgb_c.append(c2)

    knn_q_roll, hgb_q_roll = rolling(knn_q), rolling(hgb_q)
    knn_c_roll, hgb_c_roll = rolling(knn_c), rolling(hgb_c)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(knn_q_roll, label="kNN (online, judge@20%)")
    axes[0].plot(hgb_q_roll, label="HGB handcrafted (frozen, 10% train)")
    axes[0].set_title("Rolling quality (window=50)")
    axes[0].set_xlabel("requests processed"); axes[0].set_ylabel("quality"); axes[0].legend(fontsize=8)

    axes[1].plot(knn_c_roll, label="kNN (online, judge@20%)")
    axes[1].plot(hgb_c_roll, label="HGB handcrafted (frozen, 10% train)")
    axes[1].set_title("Rolling cost (window=50)")
    axes[1].set_xlabel("requests processed"); axes[1].set_ylabel("$ / request"); axes[1].legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "online_learning.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  wrote {out_path}")

    return {
        "bootstrap_pct_of_train_groups": 10,
        "judge_sample_rate": 0.2,
        "knn_final_avg_quality": sum(knn_q) / len(knn_q),
        "hgb_frozen_final_avg_quality": sum(hgb_q) / len(hgb_q),
        "knn_final_avg_cost": sum(knn_c) / len(knn_c),
        "hgb_frozen_final_avg_cost": sum(hgb_c) / len(hgb_c),
    }


# ── full-system replay (real cache + real router, via get_or_generate) ─────

def run_full_system_replay(rows, seed, cost_fn):
    shutil.rmtree(os.path.join(os.path.dirname(__file__), "..", "data"), ignore_errors=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "data"), exist_ok=True)

    from app.routing.mock_tiered_provider import MockTieredProvider
    import app.cache_engine as cache_engine_mod
    import app.context_scorer as context_scorer_mod
    from app.config import settings as app_settings
    from app.cache_engine import get_or_generate

    label_lookup = {}
    for r in rows:
        for m, v in r["labels"].items():
            label_lookup[(r["prompt"], m)] = v

    def lookup_fn(prompt, model):
        return label_lookup.get((prompt, model), {"label": 0, "quality": 0.3})

    provider = MockTieredProvider(lookup_fn, seed=seed)
    cache_engine_mod.get_provider = lambda: provider     # patch where cache_engine looks it up
    context_scorer_mod.get_provider = lambda: provider   # same instance — instant, no 0.6s sleep

    # The (unrelated) context-sensitivity scorer normally runs in a
    # background thread per miss. At this replay's throughput, dozens of
    # those threads can fire concurrently and collide writing to the same
    # sqlite3 connection ("cannot start a transaction within a
    # transaction"). Forcing it synchronous — safe here since we just
    # swapped in an instant provider above, so it adds no real latency —
    # avoids the concurrency issue entirely rather than papering over it.
    app_settings.sync_scoring = True

    # The context-sensitivity scorer always falls back to a neutral 0.5 with
    # either mock provider (neither returns valid JSON for the scorer
    # prompt) — that's the SAME documented mock-provider limitation as
    # everywhere else in this project. A neutral 0.5 never clears Layer 2's
    # THRESHOLD_FREE (0.3), and Layer 1 structurally never fires between two
    # cold-start (empty conversation_history) requests (see
    # context_engine.context_similarity's zero-vector handling) — so with
    # the real scorer, this replay's cold-start-only traffic would NEVER
    # cache-hit at all, at either layer. None of this simulated traffic
    # carries real per-user context worth protecting, so it's a fair
    # simplification (not a hidden shortcut) to score it as context-free —
    # this is exactly what a real judge would conclude for factual/task
    # queries with no prior conversation.
    cache_engine_mod.score_synchronous = lambda prompt, context_summary, answer: 0.1

    def ground_truth_fn(prompt, model):
        return lookup_fn(prompt, model)["quality"]

    rng = random.Random(seed + 2)
    stream = []
    for r in rows:
        reps = rng.choice([1, 1, 1, 2, 2, 3])
        stream.extend([r] * reps)
    rng.shuffle(stream)

    n = len(stream)
    total_cost = 0.0
    total_quality = 0.0
    hit_count = 0
    escal_count = 0
    model_counts = {}

    print(f"  replaying {n} requests (with paraphrase/exact repeats) through get_or_generate(model='auto')...")
    t0 = time.time()
    for row in stream:
        result = get_or_generate(row["prompt"], model="auto", ground_truth_fn=ground_truth_fn)
        total_cost += result.cost_usd
        served_model = result.routed_model or "mock-medium"
        model_counts[served_model] = model_counts.get(served_model, 0) + 1
        if result.cache_status == "HIT":
            hit_count += 1
        # result.escalated reflects whether the SERVED answer was originally
        # produced via a cascade escalation — on a cache HIT that's just
        # inherited history, not a new escalation happening on this
        # request. Only count it as a fresh escalation on an actual MISS,
        # where cascade_generate really ran this time.
        if result.cache_status == "MISS" and result.escalated:
            escal_count += 1
        total_quality += lookup_fn(row["prompt"], served_model)["quality"]
    elapsed = time.time() - t0

    miss_count = n - hit_count
    always_large_no_cache_cost = sum(cost_fn("mock-large", r["prompt"]) for r in stream)
    savings_pct = (
        (1 - total_cost / always_large_no_cache_cost) * 100
        if always_large_no_cache_cost > 0 else 0.0
    )

    out = {
        "n_requests": n,
        "cache_hit_rate": hit_count / n,
        "cache_miss_count": miss_count,
        "escalation_count": escal_count,
        "escalation_rate_of_all_requests": escal_count / n,
        "escalation_rate_of_misses": (escal_count / miss_count) if miss_count else 0.0,
        "routed_model_distribution": {k: v / n for k, v in model_counts.items()},
        "avg_quality": total_quality / n,
        "total_cost_usd": round(total_cost, 4),
        "always_large_no_cache_cost_usd": round(always_large_no_cache_cost, 4),
        "cost_savings_vs_always_large_no_cache_pct": round(savings_pct, 2),
        "wall_clock_seconds": round(elapsed, 1),
    }
    out_path = os.path.join(RESULTS_DIR, "full_system.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {out_path}")
    return out


# ── optional RouterBench (best-effort, never required) ─────────────────────

def try_routerbench_eval():
    try:
        from datasets import load_dataset as hf_load_dataset
        ds = hf_load_dataset("withmartian/routerbench", split="train[:200]")
        print(f"  loaded {len(ds)} RouterBench rows (best-effort router-only metrics; "
              f"skipped from the main cost-quality comparison since it has no "
              f"per-model ground truth compatible with this mock pricing/model set)")
        return {"routerbench_rows_loaded": len(ds)}
    except Exception as e:
        print(f"  RouterBench dataset unavailable ({type(e).__name__}: {e}) — skipping, as intended.")
        return None


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset", default=None, choices=[None, "routerbench"])
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.dataset == "routerbench":
        print("Attempting optional RouterBench evaluation...")
        try_routerbench_eval()

    print(f"Loading dataset from {args.data} ...")
    rows = load_dataset(args.data)
    train_rows, test_rows = split_by_group(rows, test_frac=0.2, seed=args.seed)
    print(f"  {len(rows)} total prompts -> train={len(train_rows)} test={len(test_rows)} "
          f"(split by group_id, seed={args.seed})")

    # NOTE: everything up to (not including) the full-system replay uses
    # standalone routing components (isolated ExperienceStore instances,
    # bare classifiers) — it never touches app.cache_engine's shared
    # singletons, so it can run before that module (and its data/ dir) is
    # touched at all.
    from app.embeddings import embedding_service
    from app.routing.model_registry import model_registry
    from app.routing.experience_store import ExperienceStore
    from app.routing.knn_router import KnnRouter
    from app.routing.config import RoutingSettings
    from app.routing.baseline_router import (
        HandcraftedRouter, HandcraftedPlusEmbeddingRouter, LogRegEmbeddingRouter,
    )
    from app.routing.verifier import next_tier

    cost_fn = make_cost_fn(model_registry)
    cheap_to_expensive = [m.name for m in model_registry.cheapest_to_most_expensive()]

    print("Embedding all prompts (single batched pass)...")
    t0 = time.time()
    all_texts = [r["prompt"] for r in rows]
    all_embs = embedding_service.model.encode(
        all_texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False
    ).astype("float32")
    print(f"  embedded {len(all_texts)} prompts in {time.time() - t0:.1f}s")

    emb_by_qid = {r["query_id"]: e for r, e in zip(rows, all_embs)}
    train_embs = np.stack([emb_by_qid[r["query_id"]] for r in train_rows])
    test_embs = np.stack([emb_by_qid[r["query_id"]] for r in test_rows])

    print("Fitting baseline (handcrafted-features) routers on the train split...")
    labels_per_model = {m: [r["labels"][m]["label"] for r in train_rows] for m in MODEL_NAMES}
    hgb = HandcraftedRouter(registry=model_registry)
    hgb.fit([r["prompt"] for r in train_rows], list(train_embs), labels_per_model)
    hgb_emb = HandcraftedPlusEmbeddingRouter(registry=model_registry)
    hgb_emb.fit([r["prompt"] for r in train_rows], list(train_embs), labels_per_model)
    logreg = LogRegEmbeddingRouter(registry=model_registry)
    logreg.fit([r["prompt"] for r in train_rows], list(train_embs), labels_per_model)

    print("Bootstrapping the kNN experience store from the train split...")
    eval_dir = os.path.join(os.path.dirname(__file__), "..", "data", "eval_experience_store")
    shutil.rmtree(eval_dir, ignore_errors=True)
    knn_store = ExperienceStore(data_dir=eval_dir, pass_mark=0.7)

    def full_bootstrap_rows():
        for row, emb in zip(train_rows, train_embs):
            for m in MODEL_NAMES:
                yield {
                    "query_id": row["query_id"], "embedding": emb, "prompt": row["prompt"],
                    "model": m, "quality": row["labels"][m]["quality"],
                    "cost": cost_fn(m, row["prompt"]), "latency": 100, "source": "offline",
                }
    knn_store.bootstrap_from_dataset(full_bootstrap_rows())
    knn_r = KnnRouter(registry=model_registry, store=knn_store, settings=RoutingSettings())

    rng = random.Random(args.seed)
    cascade_rng = random.Random(args.seed + 100)
    results = []

    print("Evaluating always_small / always_medium / always_large ...")
    for m in MODEL_NAMES:
        metrics = evaluate_router(fixed_route_fn(m), test_rows, test_embs, None, cost_fn, model_registry)
        results.append({"router": f"always_{model_registry.tier_of(m)}", "quality_target": "", **metrics})

    always_large_quality = next(r["avg_quality"] for r in results if r["router"] == "always_large")

    print("Evaluating oracle (cheapest correct model, hindsight) ...")
    metrics = evaluate_router(oracle_route_fn(cheap_to_expensive), test_rows, test_embs, None,
                               cost_fn, model_registry, oracle=True)
    results.append({"router": "oracle", "quality_target": "", **metrics})

    print("Evaluating random (cost-matched sweep) ...")
    for w in RANDOM_WEIGHT_SWEEP:
        metrics = evaluate_router(random_route_fn(w, rng), test_rows, test_embs, w, cost_fn, model_registry)
        results.append({"router": "random", "quality_target": w, **metrics})

    sweepable = [
        ("hgb_handcrafted", baseline_route_fn(hgb), False),
        ("hgb_handcrafted+emb", baseline_route_fn(hgb_emb), False),
        ("logreg_emb", baseline_route_fn(logreg), False),
        ("knn", knn_route_fn(knn_r), False),
        ("knn+cascade", knn_route_fn(knn_r), True),
    ]
    for name, route_fn, cascade in sweepable:
        print(f"Evaluating {name} across quality_target sweep ...")
        for qt in QT_SWEEP:
            metrics = evaluate_router(
                route_fn, test_rows, test_embs, qt, cost_fn, model_registry,
                cascade=cascade, cascade_rng=cascade_rng, next_tier_fn=next_tier,
            )
            results.append({"router": name, "quality_target": qt, **metrics})

    # ── summary.csv ────────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "summary.csv")
    fieldnames = ["router", "quality_target", "avg_quality", "avg_cost", "pct_large",
                  "severe_error_rate", "escalation_rate", "routing_p50_ms", "routing_p95_ms", "n"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in results:
            w.writerow(row)
    print(f"\nWrote {csv_path} ({len(results)} rows)")

    # ── derived per-router metrics (cost-to-95%, normalized AUC) ───────────
    by_router = {}
    for r in results:
        by_router.setdefault(r["router"], []).append((r["avg_cost"], r["avg_quality"]))

    all_costs = [c for pts in by_router.values() for c, _ in pts]
    global_min_cost, global_max_cost = min(all_costs), max(all_costs)

    derived = []
    for router, pts in by_router.items():
        derived.append({
            "router": router,
            "cost_to_reach_95pct_of_always_large": cost_to_reach_95pct(pts, always_large_quality),
            "normalized_auc": normalized_auc(pts, global_min_cost, global_max_cost),
        })
    derived_path = os.path.join(RESULTS_DIR, "derived_metrics.csv")
    with open(derived_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["router", "cost_to_reach_95pct_of_always_large", "normalized_auc"])
        w.writeheader()
        for d in derived:
            w.writerow(d)
    print(f"Wrote {derived_path}")

    # ── cost-quality curve plot ──────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    curve_routers = ["knn", "knn+cascade", "hgb_handcrafted", "hgb_handcrafted+emb", "logreg_emb", "random"]
    for router in curve_routers:
        pts = sorted(by_router.get(router, []))
        if not pts:
            continue
        xs = [c for c, _ in pts]
        ys = [q for _, q in pts]
        marker = "o-" if not router.startswith("random") else "x--"
        plt.plot(xs, ys, marker, label=router, markersize=4)

    for router in ["always_small", "always_medium", "always_large", "oracle"]:
        pts = by_router.get(router, [])
        if pts:
            c, q = pts[0]
            plt.scatter([c], [q], marker="*", s=140, label=router, zorder=5)

    plt.xlabel("avg cost per request (USD, mock pricing)")
    plt.ylabel("avg quality (success rate)")
    plt.title("Cost-quality trade-off: kNN router vs. baselines")
    plt.legend(fontsize=8, loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    curve_path = os.path.join(RESULTS_DIR, "cost_quality_curve.png")
    plt.savefig(curve_path, dpi=120)
    plt.close()
    print(f"Wrote {curve_path}")

    # ── online learning experiment ──────────────────────────────────────────
    print("\nRunning online-learning experiment (10% bootstrap, live kNN vs frozen HGB)...")
    online_results = run_online_learning_experiment(
        train_rows, train_embs, test_rows, test_embs, args.seed, model_registry, cost_fn
    )
    with open(os.path.join(RESULTS_DIR, "online_learning.json"), "w") as f:
        json.dump(online_results, f, indent=2)

    # ── full-system replay ───────────────────────────────────────────────────
    print("\nRunning full-system replay (real cache + real router via get_or_generate)...")
    full_system_results = run_full_system_replay(rows, args.seed, cost_fn)

    # ── console summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'router':22s} {'best_quality':>13s} {'cost_at_best':>13s} {'AUC':>8s} {'cost@95%large':>15s}")
    for d in sorted(derived, key=lambda x: (x["normalized_auc"] is None, -(x["normalized_auc"] or 0))):
        pts = by_router[d["router"]]
        best_c, best_q = max(pts, key=lambda p: p[1])
        auc_str = f"{d['normalized_auc']:.3f}" if d["normalized_auc"] is not None else "n/a"
        c95 = d["cost_to_reach_95pct_of_always_large"]
        c95_str = f"${c95:.6f}" if c95 is not None else "never"
        print(f"{d['router']:22s} {best_q:13.3f} {best_c:13.6f} {auc_str:>8s} {c95_str:>15s}")

    print(f"\nFull-system replay: hit_rate={full_system_results['cache_hit_rate']:.2%} "
          f"quality={full_system_results['avg_quality']:.3f} "
          f"cost_savings_vs_always_large_no_cache={full_system_results['cost_savings_vs_always_large_no_cache_pct']:.1f}%")


if __name__ == "__main__":
    main()
