"""Compare evaluation results across models.

Addresses survivorship bias by doing an outer join across metrics files
and treating missing predictions as score 0.
"""

import json
import os
from collections import defaultdict
from itertools import combinations

import fire
import pandas as pd

from config_model import EvalReport
from evaluate import get_task_code


def _model_label(path):
    return os.path.splitext(os.path.basename(path))[0]


def load_eval_results(metrics_path):
    with open(metrics_path) as f:
        data = json.load(f)
    report = EvalReport.model_validate(data)
    return {ex.id: ex for ex in report.examples}


def _build_comparison_df(results, dataset_df):
    base = dataset_df[["id", "token_length", "difficulty"]].copy()
    base["task"] = dataset_df["secondary_task"].apply(get_task_code)
    base = base.set_index("id")

    for label, scored in results.items():
        scores = pd.Series(
            {eid: ex.score for eid, ex in scored.items()},
            name=f"{label}_score",
            dtype=float,
        )
        base = base.join(scores.rename(f"{label}_score"), how="outer")

    return base


def _print_completion_matrix(df, labels):
    for la, lb in combinations(labels, 2):
        a_scored = df[f"{la}_score"].notna()
        b_scored = df[f"{lb}_score"].notna()
        both = (a_scored & b_scored).sum()
        only_a = (a_scored & ~b_scored).sum()
        only_b = (~a_scored & b_scored).sum()
        neither = (~a_scored & ~b_scored).sum()

        print(f"=== Completion: {la} vs {lb} ===")
        print(f"  Both scored:     {both:>5d}")
        print(f"  Only {la}:{' ' * max(1, 9 - len(la))}{only_a:>5d}")
        print(f"  Only {lb}:{' ' * max(1, 9 - len(lb))}{only_b:>5d}")
        print(f"  Neither scored:  {neither:>5d}")
        print(f"  Total:           {len(df):>5d}")
        print()


def _print_head_to_head(df, labels):
    for la, lb in combinations(labels, 2):
        mask = df[f"{la}_score"].notna() & df[f"{lb}_score"].notna()
        shared = df[mask]
        n = len(shared)
        if n == 0:
            print(f"=== Head-to-head: {la} vs {lb} (0 shared) ===")
            print("  No shared problems to compare.")
            print()
            continue

        a_scores = shared[f"{la}_score"]
        b_scores = shared[f"{lb}_score"]
        a_wins = (a_scores > b_scores).sum()
        ties = (a_scores == b_scores).sum()
        b_wins = (a_scores < b_scores).sum()

        print(f"=== Head-to-head: {la} vs {lb} ({n} shared) ===")
        print(f"  {la} wins:  {a_wins:>4d}  ({100 * a_wins / n:.1f}%)")
        print(f"  Ties:    {' ' * max(0, len(la) - 4)}{ties:>4d}  ({100 * ties / n:.1f}%)")
        print(f"  {lb} wins:  {b_wins:>4d}  ({100 * b_wins / n:.1f}%)")
        print()
        print(f"  Average on shared:")
        print(f"    {la}: {a_scores.mean():.3f}")
        print(f"    {lb}: {b_scores.mean():.3f}")
        print()

        task_stats = defaultdict(lambda: {"a": [], "b": []})
        for _, row in shared.iterrows():
            task_stats[row["task"]]["a"].append(row[f"{la}_score"])
            task_stats[row["task"]]["b"].append(row[f"{lb}_score"])

        print(f"  Per-task breakdown:")
        for task in sorted(task_stats.keys(), key=lambda t: (int(t[1:].split(".")[0]), int(t.split(".")[1]))):
            sa = task_stats[task]["a"]
            sb = task_stats[task]["b"]
            avg_a = sum(sa) / len(sa)
            avg_b = sum(sb) / len(sb)
            delta = avg_a - avg_b
            sign = "+" if delta >= 0 else ""
            print(f"    {task:8s} {la}={avg_a:.3f} (n={len(sa)})  {lb}={avg_b:.3f} (n={len(sb)})  delta={sign}{delta:.3f}")
        print()


def _print_adjusted_scores(df, labels, total):
    print(f"=== Adjusted overall scores ({total} total problems) ===")
    for label in labels:
        col = f"{label}_score"
        scored_mask = df[col].notna()
        n_scored = scored_mask.sum()
        raw_sum = df[col].fillna(0).sum()
        adjusted = raw_sum / total if total > 0 else 0.0
        raw_avg = df.loc[scored_mask, col].mean() if n_scored > 0 else 0.0
        print(f"  {label}: {adjusted:.3f}  (scored {n_scored}/{total}, raw avg {raw_avg:.3f})")
    print()


def _print_dimension_breakdown(df, labels, dimension, ordered_values):
    print(f"=== Adjusted scores by {dimension} ===")

    header = f"  {'':10s}"
    for label in labels:
        header += f"  {label:>24s}"
    print(header)
    print("  " + "-" * (10 + 26 * len(labels)))

    for val in ordered_values:
        subset = df[df[dimension] == val]
        n_total = len(subset)
        if n_total == 0:
            continue
        row_str = f"  {val:10s}"
        for label in labels:
            col = f"{label}_score"
            n_scored = subset[col].notna().sum()
            adj = subset[col].fillna(0).sum() / n_total if n_total > 0 else 0.0
            row_str += f"  {adj:.3f} ({n_scored:>3d}/{n_total:<3d})        "
        print(row_str.rstrip())
    print()


def compare(*metrics_files, dataset="longbench_pro_en.parquet"):
    if len(metrics_files) < 2:
        print("Error: provide at least 2 metrics JSON files")
        return

    dataset_df = pd.read_parquet(dataset)
    labels = []
    results = {}
    for path in metrics_files:
        label = _model_label(path)
        labels.append(label)
        results[label] = load_eval_results(path)
        print(f"Loaded {label}: {len(results[label])} scored examples")

    print(f"Dataset: {len(dataset_df)} problems")
    print()

    df = _build_comparison_df(results, dataset_df)

    _print_completion_matrix(df, labels)
    _print_head_to_head(df, labels)
    _print_adjusted_scores(df, labels, len(dataset_df))
    _print_dimension_breakdown(
        df, labels, "token_length",
        ["8k", "16k", "32k", "64k", "128k", "256k"],
    )
    _print_dimension_breakdown(
        df, labels, "difficulty",
        ["Easy", "Moderate", "Hard", "Extreme"],
    )


if __name__ == "__main__":
    fire.Fire(compare)
