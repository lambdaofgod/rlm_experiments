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


def _print_load_summary(results, labels, total_problems):
    print("** Load summary")
    print()
    print(f"Dataset: {total_problems} problems")
    print()
    print("| model | scored examples |")
    print("|-------+-----------------|")
    for label in labels:
        print(f"| {label} | {len(results[label])} |")
    print()


def _print_completion_matrix(df, labels):
    for la, lb in combinations(labels, 2):
        a_scored = df[f"{la}_score"].notna()
        b_scored = df[f"{lb}_score"].notna()
        both = int((a_scored & b_scored).sum())
        only_a = int((a_scored & ~b_scored).sum())
        only_b = int((~a_scored & b_scored).sum())
        neither = int((~a_scored & ~b_scored).sum())

        print(f"** Completion: {la} vs {lb}")
        print()
        print("| bucket | n |")
        print("|--------+---|")
        print(f"| Both scored | {both} |")
        print(f"| Only {la} | {only_a} |")
        print(f"| Only {lb} | {only_b} |")
        print(f"| Neither scored | {neither} |")
        print(f"| Total | {len(df)} |")
        print()


def _print_head_to_head(df, labels):
    for la, lb in combinations(labels, 2):
        mask = df[f"{la}_score"].notna() & df[f"{lb}_score"].notna()
        shared = df[mask]
        n = len(shared)
        if n == 0:
            print(f"** Head-to-head: {la} vs {lb} (0 shared)")
            print()
            print("No shared problems to compare.")
            print()
            continue

        a_scores = shared[f"{la}_score"]
        b_scores = shared[f"{lb}_score"]
        a_wins = int((a_scores > b_scores).sum())
        ties = int((a_scores == b_scores).sum())
        b_wins = int((a_scores < b_scores).sum())

        print(f"** Head-to-head: {la} vs {lb} ({n} shared)")
        print()
        print("| outcome | n | pct |")
        print("|---------+---+-----|")
        print(f"| {la} wins | {a_wins} | {100 * a_wins / n:.1f}% |")
        print(f"| Ties | {ties} | {100 * ties / n:.1f}% |")
        print(f"| {lb} wins | {b_wins} | {100 * b_wins / n:.1f}% |")
        print()
        print(f"Average on shared: {la}={a_scores.mean():.3f}, {lb}={b_scores.mean():.3f}")
        print()

        task_stats = defaultdict(lambda: {"a": [], "b": []})
        for _, row in shared.iterrows():
            task_stats[row["task"]]["a"].append(row[f"{la}_score"])
            task_stats[row["task"]]["b"].append(row[f"{lb}_score"])

        print(f"*** Per-task breakdown: {la} vs {lb}")
        print()
        print(f"| Task | {la} | n_{la} | {lb} | n_{lb} | delta |")
        print("|------+------+--------+------+--------+-------|")
        for task in sorted(task_stats.keys(), key=lambda t: (int(t[1:].split(".")[0]), int(t.split(".")[1]))):
            sa = task_stats[task]["a"]
            sb = task_stats[task]["b"]
            avg_a = sum(sa) / len(sa)
            avg_b = sum(sb) / len(sb)
            delta = avg_a - avg_b
            sign = "+" if delta >= 0 else ""
            print(f"| {task} | {avg_a:.3f} | {len(sa)} | {avg_b:.3f} | {len(sb)} | {sign}{delta:.3f} |")
        print()


def _print_shared_dimension_breakdown(df, labels, dimension, ordered_values):
    for la, lb in combinations(labels, 2):
        mask = df[f"{la}_score"].notna() & df[f"{lb}_score"].notna()
        shared = df[mask]
        if len(shared) == 0:
            continue

        print(f"*** Shared {dimension} breakdown: {la} vs {lb}")
        print()
        print(f"| {dimension} | {la} | {lb} | delta | n |")
        print("|---+---+---+---+---|")
        for val in ordered_values:
            subset = shared[shared[dimension] == val]
            n = len(subset)
            if n == 0:
                continue
            avg_a = subset[f"{la}_score"].mean()
            avg_b = subset[f"{lb}_score"].mean()
            delta = avg_a - avg_b
            sign = "+" if delta >= 0 else ""
            print(f"| {val} | {avg_a:.3f} | {avg_b:.3f} | {sign}{delta:.3f} | {n} |")
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

    df = _build_comparison_df(results, dataset_df)

    _print_load_summary(results, labels, len(dataset_df))
    _print_completion_matrix(df, labels)
    _print_head_to_head(df, labels)
    _print_shared_dimension_breakdown(
        df, labels, "token_length",
        ["8k", "16k", "32k", "64k", "128k", "256k"],
    )
    _print_shared_dimension_breakdown(
        df, labels, "difficulty",
        ["Easy", "Moderate", "Hard", "Extreme"],
    )


if __name__ == "__main__":
    fire.Fire(compare)
