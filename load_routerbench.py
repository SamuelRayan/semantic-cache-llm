"""
Downloads RouterBench (a real dataset of prompts x 11 LLMs, with real
correctness and real cost) and converts it into the long-format schema
your experience_store expects: one row per (prompt, model) pair with
a success/quality label and a cost.

Source: https://huggingface.co/datasets/withmartian/routerbench
Paper:  https://arxiv.org/abs/2403.12031

Usage:
    pip install huggingface_hub pandas
    python load_routerbench.py
"""

import json
import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "withmartian/routerbench"
FILENAME = "routerbench_0shot.pkl"   # 5-shot variant also available: routerbench_5shot.pkl
OUT_PATH = "sim/data/routerbench_routing_dataset.jsonl"


def download_and_load() -> pd.DataFrame:
    path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset")
    return pd.read_pickle(path)


def melt_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    RouterBench ships wide: one row per prompt, with per-model columns
    named '<model>|model_response', '<model>|total_cost', '<model>|performance',
    alongside shared columns like 'prompt', 'eval_name', 'sample_id'.

    This melts it to long format: one row per (prompt, model) pair —
    the shape your experience_store and baseline routers train on.
    """
    model_names = sorted({c.split("|")[0] for c in df.columns if "|" in c})

    rows = []
    for _, row in df.iterrows():
        for model in model_names:
            # The real file stores each model's 0-1 performance score under
            # the BARE model name (no suffix) — only model_response and
            # total_cost use the "<model>|<field>" convention.
            perf_col = model
            cost_col = f"{model}|total_cost"
            if perf_col not in df.columns or cost_col not in df.columns:
                continue
            perf = row.get(perf_col)
            cost = row.get(cost_col)
            if pd.isna(perf) or pd.isna(cost):
                continue
            rows.append({
                "prompt": row["prompt"],
                "eval_name": row.get("eval_name", ""),   # use as a rough group proxy —
                "sample_id": row.get("sample_id", ""),   # no true paraphrase clusters here
                "model": model,
                "quality_score": float(perf),            # 0.0-1.0, often binary
                "success": bool(float(perf) >= 0.5),
                "cost_usd": float(cost),
            })
    return pd.DataFrame(rows)


def assign_tiers(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps RouterBench's 11 real models onto your project's small/medium/large
    tiers by their real average cost, so this plugs straight into your
    existing model_registry without any code changes there.
    """
    avg_cost = long_df.groupby("model")["cost_usd"].mean().sort_values()
    models_sorted = avg_cost.index.tolist()
    n = len(models_sorted)
    tier_map = {}
    for i, m in enumerate(models_sorted):
        if i < n // 3:
            tier_map[m] = "small"
        elif i < 2 * n // 3:
            tier_map[m] = "medium"
        else:
            tier_map[m] = "large"

    long_df["tier"] = long_df["model"].map(tier_map)

    print("Model -> tier mapping (by real average cost):")
    for m in models_sorted:
        succ = long_df.loc[long_df["model"] == m, "success"].mean()
        print(f"  {m:32s} avg_cost=${avg_cost[m]:.5f}  success_rate={succ:.3f}  tier={tier_map[m]}")

    return long_df


def main():
    print(f"Downloading {FILENAME} from {REPO_ID} ...")
    df = download_and_load()
    n_models = len([c for c in df.columns if "|" in c]) // 3
    print(f"Loaded {len(df)} prompts x {n_models} models (wide format)\n")

    long_df = melt_to_long(df)
    long_df = assign_tiers(long_df)

    long_df.to_json(OUT_PATH, orient="records", lines=True)

    print(f"\nWrote {len(long_df)} (prompt, model) rows to {OUT_PATH}")
    print(f"Unique prompts: {long_df['prompt'].nunique()}")
    print(f"Unique models:  {long_df['model'].nunique()}")
    print(f"Overall success rate: {long_df['success'].mean():.3f}")
    print(f"Eval categories (rough grouping, not paraphrase clusters): "
          f"{long_df['eval_name'].nunique()}")


if __name__ == "__main__":
    main()
