"""Real-data validation track: evaluates the SAME router implementations
already built in app/routing/ (kNN, HGB baselines, logreg) against
RouterBench — ~36k real prompts x up to 11 real LLMs, real correctness,
real cost. Completely isolated from the synthetic track: separate data
directory, separate script, never touches the production cache files or
the synthetic eval's own experience store.

No LLM calls anywhere — RouterBench's outcomes are pre-generated.

Usage (from the project root):
    python eval/run_routing_eval_real.py

Outputs (eval/results/):
    real_summary.csv
    real_cost_quality_curve.png
    real_per_category.csv
    synthetic_vs_real_comparison.csv
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import yaml

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "sim", "data", "routerbench_routing_dataset.jsonl")
REAL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "routing_real")

REAL_QT_SWEEP = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
RANDOM_WEIGHT_SWEEP = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
CASCADE_CONFIDENCE_THRESHOLD = 0.5   # see note in evaluate_router / report


# ── loading ──────────────────────────────────────────────────────────────────

def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_prompt_meta_and_lookup(rows):
    """Returns (prompts_meta, outcome_lookup, model_names).
    prompts_meta: list of dicts {sample_id, prompt, eval_name}, one per
    unique sample_id. outcome_lookup[sample_id][model] = {success, quality, cost}."""
    outcome_lookup = defaultdict(dict)
    meta_by_id = {}
    for r in rows:
        sid = r["sample_id"]
        outcome_lookup[sid][r["model"]] = {
            "success": bool(r["success"]), "quality": float(r["quality_score"]), "cost": float(r["cost_usd"]),
        }
        if sid not in meta_by_id:
            meta_by_id[sid] = {"sample_id": sid, "prompt": r["prompt"], "eval_name": r["eval_name"]}
    model_names = sorted({r["model"] for r in rows})
    prompts_meta = list(meta_by_id.values())
    return prompts_meta, outcome_lookup, model_names


def filter_full_coverage(prompts_meta, outcome_lookup, model_names):
    kept = [p for p in prompts_meta if set(outcome_lookup[p["sample_id"]].keys()) >= set(model_names)]
    dropped = len(prompts_meta) - len(kept)
    print(f"  full-coverage filter: kept {len(kept)}/{len(prompts_meta)} prompts "
          f"({dropped} dropped for missing outcomes on at least one of {len(model_names)} models)")
    return kept


# ── stratified split by eval_name (prompt-level, not row-level) ────────────

def stratified_split(prompts_meta, test_frac=0.2, seed=42):
    by_cat = defaultdict(list)
    for p in prompts_meta:
        by_cat[p["eval_name"]].append(p)

    rng = random.Random(seed)
    train, test = [], []
    print(f"  {'category':40s} {'total':>8s} {'train':>8s} {'test':>8s}")
    for cat in sorted(by_cat):
        items = by_cat[cat][:]
        rng.shuffle(items)
        n_test = max(1, round(len(items) * test_frac)) if len(items) > 1 else 0
        cat_test = items[:n_test]
        cat_train = items[n_test:]
        train.extend(cat_train)
        test.extend(cat_test)
        print(f"  {cat[:40]:40s} {len(items):8d} {len(cat_train):8d} {len(cat_test):8d}")
    return train, test


# ── real model registry (dynamically built from the dataset itself) ────────

def build_real_registry(rows, out_yaml_path):
    from app.routing.model_registry import ModelRegistry

    costs = defaultdict(list)
    tiers = {}
    for r in rows:
        costs[r["model"]].append(r["cost_usd"])
        tiers[r["model"]] = r["tier"]

    models = []
    for m, cs in costs.items():
        avg_cost = sum(cs) / len(cs)
        models.append({
            "name": m, "tier": tiers[m],
            # Real RouterBench cost is a single per-inference total (not
            # split input/output), so it's used directly as the ranking
            # proxy here — cost_per_1k_output stays 0. Actual per-request
            # cost accounting below uses each row's REAL recorded cost_usd
            # directly, never this registry's derived rate.
            "cost_per_1k_input": avg_cost, "cost_per_1k_output": 0.0,
            "avg_latency_ms": 500, "enabled": True,
        })
    os.makedirs(os.path.dirname(out_yaml_path), exist_ok=True)
    with open(out_yaml_path, "w") as f:
        yaml.safe_dump({"models": models}, f)
    return ModelRegistry(config_path=out_yaml_path)


# ── embeddings (cached to disk) ─────────────────────────────────────────────

def get_or_build_embeddings(prompts_meta):
    from app.embeddings import embedding_service

    emb_path = os.path.join(REAL_DIR, "embeddings.npy")
    ids_path = os.path.join(REAL_DIR, "embedding_ids.json")
    os.makedirs(REAL_DIR, exist_ok=True)

    wanted_ids = [p["sample_id"] for p in prompts_meta]

    if os.path.exists(emb_path) and os.path.exists(ids_path):
        with open(ids_path) as f:
            cached_ids = json.load(f)
        if cached_ids == wanted_ids:
            print(f"  loaded cached embeddings from {emb_path} (no re-embedding)")
            return np.load(emb_path)
        print("  cached embeddings don't match current prompt set — re-embedding")

    print(f"  embedding {len(prompts_meta)} unique prompts (single batched pass)...")
    t0 = time.time()
    texts = [p["prompt"] for p in prompts_meta]
    embs = embedding_service.model.encode(
        texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False
    ).astype("float32")
    print(f"  embedded in {time.time() - t0:.1f}s")

    np.save(emb_path, embs)
    with open(ids_path, "w") as f:
        json.dump(wanted_ids, f)
    return embs


# ── router adapters (same interface as the synthetic track) ────────────────

def fixed_route_fn(model_name):
    def route(prompt, emb, qt):
        return model_name, {"reason": "fixed", "routing_ms": 0.0}
    return route


def oracle_route_fn():
    def route(prompt, emb, qt, outcomes=None):
        # cheapest REAL model (by this prompt's own real cost) that actually succeeded
        correct = [(o["cost"], m) for m, o in outcomes.items() if o["success"]]
        if not correct:
            # nobody succeeded — fall back to the cheapest attempt anyway
            cheapest = min(outcomes.items(), key=lambda kv: kv[1]["cost"])
            return cheapest[0], {"reason": "oracle_none_succeed", "routing_ms": 0.0}
        correct.sort(key=lambda x: x[0])
        return correct[0][1], {"reason": "oracle", "routing_ms": 0.0}
    return route


def random_route_fn(large_weight, rng, registry):
    def route(prompt, emb, qt):
        t0 = time.perf_counter()
        r = rng.random()
        tier = "large" if r < large_weight else ("medium" if r < large_weight + (1 - large_weight) / 2 else "small")
        candidates = registry.by_tier(tier) or registry.enabled_models()
        chosen = rng.choice(candidates).name
        return chosen, {"reason": "random", "routing_ms": (time.perf_counter() - t0) * 1000}
    return route


def knn_route_fn(knn_router_obj):
    def route(prompt, emb, qt):
        d = knn_router_obj.route(prompt, emb, quality_target=qt)
        return d.model, {"reason": d.reason, "routing_ms": d.routing_ms, "p_hat": d.p_hat}
    return route


def baseline_route_fn(baseline_obj):
    def route(prompt, emb, qt):
        d = baseline_obj.route(prompt, emb, quality_target=qt)
        return d.model, {"reason": d.reason, "routing_ms": d.routing_ms}
    return route


# ── evaluation ───────────────────────────────────────────────────────────────

def evaluate_router(route_fn, test_meta, test_embs, outcome_lookup, qt, registry,
                     cascade=False, oracle=False, next_tier_fn=None):
    n = len(test_meta)
    total_quality = total_cost = 0.0
    large_count = severe_errors = escalations = 0
    latencies = []

    cheap_to_expensive_names = [m.name for m in registry.cheapest_to_most_expensive()]

    for meta, emb in zip(test_meta, test_embs):
        outcomes = outcome_lookup[meta["sample_id"]]

        if oracle:
            chosen, info = route_fn(meta["prompt"], emb, qt, outcomes=outcomes)
        else:
            chosen, info = route_fn(meta["prompt"], emb, qt)
        latencies.append(info.get("routing_ms", 0.0))
        orig_chosen = chosen

        if cascade:
            # RouterBench has no finish_reason/logprobs — there is no live
            # generation to inspect the surface of. So instead of
            # response-shape verification (the synthetic track's approach),
            # this cascade is CONFIDENCE-gated: escalate one tier whenever
            # the router's OWN p_hat for its chosen model is below
            # CASCADE_CONFIDENCE_THRESHOLD, decided at routing time, before
            # the (frozen, pre-recorded) outcome is even looked up. This is
            # a materially weaker signal than a real verifier — documented
            # explicitly in ROUTING_REPORT.md.
            p_hat_chosen = info.get("p_hat", {}).get(chosen)
            if p_hat_chosen is not None and p_hat_chosen < CASCADE_CONFIDENCE_THRESHOLD:
                tier = registry.tier_of(chosen)
                nxt = next_tier_fn(tier)
                if nxt:
                    candidates = registry.by_tier(nxt)
                    if candidates:
                        esc_model = min(candidates, key=lambda m: outcomes.get(m.name, {}).get("cost", 1e9)).name
                        if esc_model in outcomes:
                            escalations += 1
                            chosen = esc_model

        outcome = outcomes.get(chosen)
        if outcome is None:
            continue  # shouldn't happen post full-coverage filter, but stay safe
        quality = outcome["quality"]
        cost = outcome["cost"]

        total_quality += quality
        total_cost += cost
        if registry.tier_of(chosen) == "large":
            large_count += 1

        only_large_succeeds = (
            any(registry.tier_of(m) == "large" and o["success"] for m, o in outcomes.items())
            and not any(registry.tier_of(m) == "small" and o["success"] for m, o in outcomes.items())
            and not any(registry.tier_of(m) == "medium" and o["success"] for m, o in outcomes.items())
        )
        if only_large_succeeds and registry.tier_of(orig_chosen) == "small":
            severe_errors += 1

    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else 0.0
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0.0

    return {
        "avg_quality": total_quality / n, "avg_cost": total_cost / n,
        "pct_large": large_count / n, "severe_error_rate": severe_errors / n,
        "escalation_rate": escalations / n, "routing_p50_ms": p50, "routing_p95_ms": p95, "n": n,
    }


def normalized_auc(points, gmin, gmax, n_samples=200):
    if len(points) < 2 or gmax <= gmin:
        return None
    pts = sorted(points, key=lambda p: p[0])
    xs = np.array([c for c, _ in pts], dtype="float64")
    ys = np.array([q for _, q in pts], dtype="float64")
    grid = np.linspace(gmin, gmax, n_samples)
    interp_y = np.interp(grid, xs, ys, left=ys[0], right=ys[-1])
    return float(np.trapz(interp_y, grid) / (gmax - gmin))


def cost_to_reach_95pct(points, baseline_quality):
    target = 0.95 * baseline_quality
    qualifying = [c for c, q in points if q >= target]
    return min(qualifying) if qualifying else None


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    global REAL_DIR, RESULTS_DIR

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-categories", type=int, default=None,
                     help="Smoke-test only: limit to the first N eval_name categories (sorted).")
    ap.add_argument("--max-prompts-per-category", type=int, default=None,
                     help="Smoke-test only: cap prompts kept per category before splitting.")
    ap.add_argument("--real-dir", default=None,
                     help="Override the isolated data dir (default data/routing_real) — "
                          "use a scratch dir for smoke tests so they never touch the real run's cache.")
    ap.add_argument("--results-dir", default=None,
                     help="Override the results output dir — use a scratch dir for smoke tests.")
    args = ap.parse_args()

    if args.real_dir:
        REAL_DIR = args.real_dir
    if args.results_dir:
        RESULTS_DIR = args.results_dir

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REAL_DIR, exist_ok=True)

    print(f"Loading RouterBench dataset from {args.data} ...")
    rows = load_rows(args.data)
    prompts_meta, outcome_lookup, model_names = build_prompt_meta_and_lookup(rows)
    print(f"  {len(rows)} (prompt, model) rows, {len(prompts_meta)} unique prompts, "
          f"{len(model_names)} models, {len(set(p['eval_name'] for p in prompts_meta))} categories")

    if args.max_categories or args.max_prompts_per_category:
        cats = sorted(set(p["eval_name"] for p in prompts_meta))
        if args.max_categories:
            cats = cats[:args.max_categories]
        keep_cats = set(cats)
        by_cat_count = defaultdict(int)
        limited = []
        for p in prompts_meta:
            if p["eval_name"] not in keep_cats:
                continue
            if args.max_prompts_per_category and by_cat_count[p["eval_name"]] >= args.max_prompts_per_category:
                continue
            by_cat_count[p["eval_name"]] += 1
            limited.append(p)
        prompts_meta = limited
        kept_ids = {p["sample_id"] for p in prompts_meta}
        rows = [r for r in rows if r["sample_id"] in kept_ids]
        print(f"  SMOKE-TEST SUBSET: limited to {len(cats)} categories, "
              f"{len(prompts_meta)} prompts, {len(rows)} rows")

    prompts_meta = filter_full_coverage(prompts_meta, outcome_lookup, model_names)

    print("Stratified split by eval_name (seed=42, 80/20, prompt-level)...")
    train_meta, test_meta = stratified_split(prompts_meta, test_frac=0.2, seed=args.seed)
    print(f"  TOTAL: train={len(train_meta)} test={len(test_meta)}")

    print("Building the real model registry (dynamic, from the dataset's own costs/tiers)...")
    registry_yaml = os.path.join(REAL_DIR, "models_real.yaml")
    registry = build_real_registry(rows, registry_yaml)
    for m in registry.cheapest_to_most_expensive():
        print(f"  {m.name:35s} tier={m.tier:7s} avg_cost=${m.cost_per_1k_input:.5f}")

    print("Embedding prompts (cached to data/routing_real/)...")
    all_embs = get_or_build_embeddings(prompts_meta)
    emb_by_id = {p["sample_id"]: e for p, e in zip(prompts_meta, all_embs)}
    train_embs = np.stack([emb_by_id[p["sample_id"]] for p in train_meta])
    test_embs = np.stack([emb_by_id[p["sample_id"]] for p in test_meta])

    print("Bootstrapping the REAL (isolated) kNN experience store from the train split...")
    from app.routing.experience_store import ExperienceStore
    from app.routing.knn_router import KnnRouter
    from app.routing.config import RoutingSettings
    from app.routing.baseline_router import (
        HandcraftedRouter, HandcraftedPlusEmbeddingRouter, LogRegEmbeddingRouter,
    )
    from app.routing.verifier import next_tier

    store_dir = os.path.join(REAL_DIR, "experience_store")
    real_store = ExperienceStore(data_dir=store_dir, pass_mark=0.5)   # RouterBench's own success cutoff

    def bootstrap_rows():
        for meta, emb in zip(train_meta, train_embs):
            outcomes = outcome_lookup[meta["sample_id"]]
            for model, o in outcomes.items():
                yield {
                    "query_id": meta["sample_id"], "embedding": emb, "prompt": meta["prompt"],
                    "model": model, "quality": o["quality"], "cost": o["cost"],
                    "latency": 500, "source": "offline",
                }
    t0 = time.time()
    real_store.bootstrap_from_dataset(bootstrap_rows())
    print(f"  bootstrapped {real_store.count()} outcomes over {real_store.unique_query_count()} "
          f"unique queries in {time.time() - t0:.1f}s")

    knn_settings = RoutingSettings(pass_mark=0.5)
    knn_r = KnnRouter(registry=registry, store=real_store, settings=knn_settings)

    print("Fitting baseline (handcrafted-features) routers on the real train split...")
    labels_per_model = {m: [] for m in model_names}
    for meta in train_meta:
        outcomes = outcome_lookup[meta["sample_id"]]
        for m in model_names:
            labels_per_model[m].append(1 if outcomes[m]["success"] else 0)

    t0 = time.time()
    hgb = HandcraftedRouter(registry=registry)
    hgb.fit([p["prompt"] for p in train_meta], list(train_embs), labels_per_model)
    hgb_emb = HandcraftedPlusEmbeddingRouter(registry=registry)
    hgb_emb.fit([p["prompt"] for p in train_meta], list(train_embs), labels_per_model)
    logreg = LogRegEmbeddingRouter(registry=registry)
    logreg.fit([p["prompt"] for p in train_meta], list(train_embs), labels_per_model)
    print(f"  fit 3 baseline router variants ({len(model_names)} classifiers each) in {time.time() - t0:.1f}s")

    rng = random.Random(args.seed)
    results = []

    print("Evaluating always_small / always_medium / always_large ...")
    for tier in ["small", "medium", "large"]:
        candidates = registry.by_tier(tier)
        cheapest = min(candidates, key=lambda m: m.cost_per_1k_input)
        metrics = evaluate_router(fixed_route_fn(cheapest.name), test_meta, test_embs,
                                   outcome_lookup, None, registry)
        results.append({"router": f"always_{tier}", "quality_target": "", **metrics})

    always_large_quality = next(r["avg_quality"] for r in results if r["router"] == "always_large")

    print("Evaluating oracle (cheapest correct model, hindsight, real costs) ...")
    metrics = evaluate_router(oracle_route_fn(), test_meta, test_embs, outcome_lookup, None,
                               registry, oracle=True)
    results.append({"router": "oracle", "quality_target": "", **metrics})

    print("Evaluating random (cost-matched sweep) ...")
    for w in RANDOM_WEIGHT_SWEEP:
        metrics = evaluate_router(random_route_fn(w, rng, registry), test_meta, test_embs,
                                   outcome_lookup, w, registry)
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
        t0 = time.time()
        for qt in REAL_QT_SWEEP:
            metrics = evaluate_router(route_fn, test_meta, test_embs, outcome_lookup, qt,
                                       registry, cascade=cascade, next_tier_fn=next_tier)
            results.append({"router": name, "quality_target": qt, **metrics})
        print(f"  done in {time.time() - t0:.1f}s")

    # ── real_summary.csv ─────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "real_summary.csv")
    fieldnames = ["router", "quality_target", "avg_quality", "avg_cost", "pct_large",
                  "severe_error_rate", "escalation_rate", "routing_p50_ms", "routing_p95_ms", "n"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in results:
            w.writerow(row)
    print(f"\nWrote {csv_path} ({len(results)} rows)")

    by_router = defaultdict(list)
    for r in results:
        by_router[r["router"]].append((r["avg_cost"], r["avg_quality"]))
    all_costs = [c for pts in by_router.values() for c, _ in pts]
    gmin, gmax = min(all_costs), max(all_costs)

    derived = {}
    for router, pts in by_router.items():
        derived[router] = {
            "cost_to_reach_95pct_of_always_large": cost_to_reach_95pct(pts, always_large_quality),
            "normalized_auc": normalized_auc(pts, gmin, gmax),
        }

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    for router in ["knn", "knn+cascade", "hgb_handcrafted", "hgb_handcrafted+emb", "logreg_emb", "random"]:
        pts = sorted(by_router.get(router, []))
        if not pts:
            continue
        xs, ys = zip(*pts)
        marker = "x--" if router == "random" else "o-"
        plt.plot(xs, ys, marker, label=router, markersize=4)
    for router in ["always_small", "always_medium", "always_large", "oracle"]:
        pts = by_router.get(router, [])
        if pts:
            c, q = pts[0]
            plt.scatter([c], [q], marker="*", s=140, label=router, zorder=5)
    plt.xlabel("avg cost per request (USD, real RouterBench pricing)")
    plt.ylabel("avg quality (success rate)")
    plt.title("Cost-quality trade-off on real data (RouterBench)")
    plt.legend(fontsize=8, loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    curve_path = os.path.join(RESULTS_DIR, "real_cost_quality_curve.png")
    plt.savefig(curve_path, dpi=120)
    plt.close()
    print(f"Wrote {curve_path}")

    # ── per-category breakdown ───────────────────────────────────────────────
    print("Computing per-category breakdown (knn+cascade, hgb_handcrafted+emb, always_large)...")
    test_by_cat = defaultdict(list)
    for meta, emb in zip(test_meta, test_embs):
        test_by_cat[meta["eval_name"]].append((meta, emb))

    cat_rows = []
    per_cat_routers = [
        ("always_large", fixed_route_fn(min(registry.by_tier("large"), key=lambda m: m.cost_per_1k_input).name), False, False),
        ("hgb_handcrafted+emb", baseline_route_fn(hgb_emb), False, False),
        ("knn+cascade", knn_route_fn(knn_r), True, False),
    ]
    for cat, items in sorted(test_by_cat.items()):
        cat_meta = [m for m, _ in items]
        cat_embs = np.stack([e for _, e in items])
        for name, route_fn, cascade, oracle in per_cat_routers:
            pts = []
            for qt in (REAL_QT_SWEEP if name != "always_large" else [None]):
                m = evaluate_router(route_fn, cat_meta, cat_embs, outcome_lookup, qt,
                                     registry, cascade=cascade, next_tier_fn=next_tier)
                pts.append((m["avg_cost"], m["avg_quality"]))
            auc = normalized_auc(pts, gmin, gmax) if len(pts) > 1 else None
            best_q = max(q for _, q in pts)
            cat_rows.append({
                "eval_name": cat, "router": name, "n_test": len(cat_meta),
                "best_quality": best_q, "normalized_auc": auc,
            })
    per_cat_path = os.path.join(RESULTS_DIR, "real_per_category.csv")
    with open(per_cat_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["eval_name", "router", "n_test", "best_quality", "normalized_auc"])
        w.writeheader()
        for r in cat_rows:
            w.writerow(r)
    print(f"Wrote {per_cat_path} ({len(cat_rows)} rows across {len(test_by_cat)} categories)")

    # ── synthetic-vs-real comparison ─────────────────────────────────────────
    print("Cross-checking against the synthetic-data verdict (eval/results/derived_metrics.csv)...")
    synth_path = os.path.join(RESULTS_DIR, "derived_metrics.csv")
    synth_auc = {}
    if os.path.exists(synth_path):
        with open(synth_path) as f:
            for row in csv.DictReader(f):
                if row["normalized_auc"]:
                    synth_auc[row["router"]] = float(row["normalized_auc"])
    else:
        print(f"  WARNING: {synth_path} not found — run the synthetic eval first for a full comparison")

    compare_routers = ["knn", "knn+cascade", "hgb_handcrafted", "hgb_handcrafted+emb", "logreg_emb", "random"]
    comparison_rows = []
    for router in compare_routers:
        comparison_rows.append({
            "router": router,
            "synthetic_auc": synth_auc.get(router),
            "real_auc": derived.get(router, {}).get("normalized_auc"),
        })
    comp_path = os.path.join(RESULTS_DIR, "synthetic_vs_real_comparison.csv")
    with open(comp_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["router", "synthetic_auc", "real_auc"])
        w.writeheader()
        for r in comparison_rows:
            w.writerow(r)
    print(f"Wrote {comp_path}")

    spearman_rho = None
    paired = [(r["synthetic_auc"], r["real_auc"]) for r in comparison_rows
              if r["synthetic_auc"] is not None and r["real_auc"] is not None]
    if len(paired) >= 3:
        from scipy.stats import spearmanr
        synth_vals = [p[0] for p in paired]
        real_vals = [p[1] for p in paired]
        spearman_rho, spearman_p = spearmanr(synth_vals, real_vals)
        print(f"  Spearman rank correlation (synthetic AUC vs real AUC, n={len(paired)}): "
              f"rho={spearman_rho:.3f}  p={spearman_p:.3f}")
    else:
        print("  not enough paired routers with both AUCs to compute Spearman correlation")

    # ── console summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("REAL-DATA SUMMARY")
    print("=" * 78)
    print(f"{'router':22s} {'best_quality':>13s} {'cost_at_best':>14s} {'AUC':>8s} {'cost@95%large':>15s}")
    for router, pts in sorted(by_router.items(), key=lambda kv: -(derived[kv[0]]["normalized_auc"] or -1)):
        d = derived[router]
        best_c, best_q = max(pts, key=lambda p: p[1])
        auc_str = f"{d['normalized_auc']:.3f}" if d["normalized_auc"] is not None else "n/a"
        c95 = d["cost_to_reach_95pct_of_always_large"]
        c95_str = f"${c95:.6f}" if c95 is not None else "never"
        print(f"{router:22s} {best_q:13.3f} {best_c:14.6f} {auc_str:>8s} {c95_str:>15s}")

    if spearman_rho is not None:
        print(f"\nSpearman(synthetic AUC, real AUC) over {len(paired)} routers: rho={spearman_rho:.3f}")


if __name__ == "__main__":
    main()
