"""Evaluate predictions against LongBench-Pro gold answers.

Uses the correct metric per secondary_task as defined by LongBench-Pro.
"""

import json
import math
import re
from collections import defaultdict

import fire
import pandas as pd

from config_model import EvalReport, EvalSummary, GroupStats, ScoredExample, load_config


# ---------------------------------------------------------------------------
# Answer normalization
# ---------------------------------------------------------------------------

def normalize_element(s):
    """Lowercase, strip whitespace from a single answer element."""
    return str(s).strip().lower()


def parse_answer_list(raw):
    """Parse a prediction or gold answer into a list of normalized strings.

    Handles:
    - JSON-encoded lists: '["a", "b"]'
    - Python list repr: "['a', 'b']"
    - Newline-separated values
    - Single value: "42"
    """
    if isinstance(raw, list):
        return [normalize_element(x) for x in raw]

    s = str(raw).strip()

    # Try JSON first
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [normalize_element(x) for x in parsed]
        return [normalize_element(parsed)]
    except (json.JSONDecodeError, TypeError):
        pass

    # Try Python list repr (e.g. "['a', 'b']")
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1]
        parts = [p.strip().strip("'\"") for p in inner.split(",") if p.strip()]
        if parts:
            return [normalize_element(x) for x in parts]

    # Newline-separated (LongBench-Pro default format)
    if "\n" in s:
        parts = [p.strip() for p in s.split("\n") if p.strip()]
        if parts:
            return [normalize_element(x) for x in parts]

    # Single value
    return [normalize_element(s)]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def accuracy(gold, pred):
    """Exact match on first element only."""
    if not gold or not pred:
        return 0.0
    return 1.0 if gold[0] == pred[0] else 0.0


def sub_em(gold, pred):
    """Subset exact match: fraction of gold elements found in prediction."""
    if not gold:
        return 1.0
    pred_set = set(pred)
    matched = sum(1 for g in gold if g in pred_set)
    return matched / len(gold)


def f1_score(gold, pred):
    """Set-level F1 over answer elements."""
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    gold_set = set(gold)
    pred_set = set(pred)
    intersection = gold_set & pred_set
    if not intersection:
        return 0.0
    precision = len(intersection) / len(pred_set)
    recall = len(intersection) / len(gold_set)
    return 2 * precision * recall / (precision + recall)


def pairwise_accuracy(gold, pred):
    """Fraction of gold item pairs that appear in correct relative order in pred."""
    if len(gold) < 2:
        return 1.0

    pred_pos = {}
    for i, p in enumerate(pred):
        if p not in pred_pos:
            pred_pos[p] = i

    total_pairs = 0
    correct_pairs = 0
    for i in range(len(gold)):
        for j in range(i + 1, len(gold)):
            if gold[i] in pred_pos and gold[j] in pred_pos:
                total_pairs += 1
                if pred_pos[gold[i]] < pred_pos[gold[j]]:
                    correct_pairs += 1

    if total_pairs == 0:
        return 0.0
    return correct_pairs / total_pairs


def _dcg(relevances):
    """Discounted cumulative gain."""
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg(gold, pred):
    """NDCG@k where k = len(gold).

    Gold items are assigned decreasing relevance by position.
    """
    k = len(gold)
    if k == 0:
        return 1.0

    gold_rel = {}
    for i, g in enumerate(gold):
        if g not in gold_rel:
            gold_rel[g] = k - i

    pred_rels = []
    for p in pred[:k]:
        pred_rels.append(gold_rel.get(p, 0))
    pred_rels.extend([0] * (k - len(pred_rels)))

    dcg_val = _dcg(pred_rels)
    ideal_rels = sorted(gold_rel.values(), reverse=True)[:k]
    ideal_rels.extend([0] * (k - len(ideal_rels)))
    idcg_val = _dcg(ideal_rels)

    if idcg_val == 0:
        return 0.0
    return dcg_val / idcg_val


def summary_score(_gold, _pred):
    """Stub for summary metric (requires embedding model). Returns NaN."""
    return float("nan")


# ---------------------------------------------------------------------------
# Metric dispatch
# ---------------------------------------------------------------------------

METRIC_MAP = {
    "T1.1": ndcg,
    "T1.2": ndcg,
    "T2.1": pairwise_accuracy,
    "T2.2": pairwise_accuracy,
    "T3.1": accuracy,
    "T3.2": accuracy,
    "T4.1": summary_score,
    "T4.2": summary_score,
    "T5.1": f1_score,
    "T5.2": f1_score,
    "T6.1": sub_em,
    "T6.2": f1_score,
    "T6.3": pairwise_accuracy,
    "T7.1": f1_score,
    "T7.2": f1_score,
    "T7.3": f1_score,
    "T8.1": sub_em,
    "T8.2": sub_em,
    "T8.3": sub_em,
    "T9.1": f1_score,
    "T9.2": f1_score,
    "T10.1": sub_em,
    "T10.2": sub_em,
    "T11.1": accuracy,
    "T11.2": accuracy,
}


def get_task_code(secondary_task):
    """Extract task code like 'T1.1' from 'T1.1 Global Cohesive Retrieval'."""
    match = re.match(r"(T\d+\.\d+)", secondary_task)
    return match.group(1) if match else secondary_task


def get_metric(secondary_task):
    code = get_task_code(secondary_task)
    return METRIC_MAP.get(code)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    predictions_path,
    dataset_path="longbench_pro_en.parquet",
    pred_col="pred_answer",
    id_col="id",
    show_errors=False,
    output_metrics=None,
):
    """Evaluate predictions against gold answers.

    Expects a CSV/parquet with at least `id` and `pred_answer` columns.
    Joins against the dataset on `id` to get gold answers and secondary_task.
    """
    df_gold = pd.read_parquet(dataset_path)

    if predictions_path.endswith(".parquet"):
        df_pred = pd.read_parquet(predictions_path)
    else:
        df_pred = pd.read_csv(predictions_path)

    # Normalize column names to match _score_predictions expectations
    df_pred = df_pred.rename(columns={id_col: "id", pred_col: "pred_answer"})
    _score_predictions(df_gold, df_pred, show_errors=show_errors, output_metrics=output_metrics)


# ---------------------------------------------------------------------------
# Phoenix trace-based evaluation
# ---------------------------------------------------------------------------

def _extract_answer_from_prediction_repr(output_json_str):
    """Extract answer string from RLM span output.

    Output is a JSON list where element 0 is a Prediction repr like:
      Prediction(
          answer='...VALUE...',
          trajectory=[...])

    The answer field can be single-quoted or double-quoted.
    """
    parsed = json.loads(output_json_str)
    repr_str = parsed[0]

    # Try both quoting styles: answer='...' and answer="..."
    for quote in ("'", '"'):
        sep = f"{quote},\n    trajectory="
        prefix = f"Prediction(\n    answer={quote}"
        if prefix in repr_str and sep in repr_str:
            answer_part = repr_str.split(sep)[0]
            answer_part = answer_part.replace(prefix, "", 1)
            answer_part = answer_part.replace("\\n", "\n")
            # Strip [Answer] prefix leak from LongBench-Pro format
            answer_part = re.sub(r"^\[Answer\]\s*", "", answer_part)
            return answer_part

    raise ValueError(f"Cannot parse Prediction repr: {repr_str[:200]}")


def fetch_predictions(backend, project_name, dataset_path, limit=10000):
    """Fetch RLM predictions from traces and match to dataset rows.

    HACK: Matching is done by comparing span query text to dataset
    question_nonthinking. Tries direct match first, then applies
    transform_question for pre-transform spans. This is fragile and needs
    a systematic fix (e.g. passing row id through to spans).
    """
    from data_utils import transform_question
    from tracing_backend import INPUT_VALUE, NAME, OUTPUT_VALUE, START_TIME, STATUS_CODE

    spans_df = backend.get_root_spans(project_name, limit=limit)

    # Filter to successful RLM forward spans only
    rlm_spans = spans_df[spans_df[NAME] == "RLM.forward"]
    ok_spans = rlm_spans[rlm_spans[STATUS_CODE] == "OK"].copy()
    print(f"Fetched {len(spans_df)} root spans, {len(rlm_spans)} RLM.forward, {len(ok_spans)} OK")

    # Load dataset for matching
    dataset = pd.read_parquet(dataset_path)

    # Build lookup: question_nonthinking -> row
    question_to_row = {}
    for _, row in dataset.iterrows():
        question_to_row[row["question_nonthinking"]] = row

    # Also build transformed-question lookup for pre-transform spans
    transformed_to_row = {}
    for _, row in dataset.iterrows():
        transformed = transform_question(row["question_nonthinking"])
        transformed_to_row[transformed] = row

    results = []
    unmatched = 0
    for _, span in ok_spans.iterrows():
        input_data = json.loads(span[INPUT_VALUE])
        query = input_data["input_args"]["query"]

        # Try direct match first (post-transform spans)
        matched_row = question_to_row.get(query)

        # Fallback: apply transform_question (pre-transform spans)
        if matched_row is None:
            transformed_query = transform_question(query)
            matched_row = question_to_row.get(transformed_query)

        # Also try matching against the transformed lookup
        if matched_row is None:
            matched_row = transformed_to_row.get(query)

        if matched_row is None:
            unmatched += 1
            continue

        try:
            answer = _extract_answer_from_prediction_repr(span[OUTPUT_VALUE])
        except (json.JSONDecodeError, IndexError, KeyError, ValueError):
            unmatched += 1
            continue

        results.append({
            "id": matched_row["id"],
            "question_nonthinking": matched_row["question_nonthinking"],
            "pred_answer": answer,
            "start_time": span[START_TIME],
        })

    if unmatched:
        print(f"Warning: {unmatched} spans could not be matched to dataset rows")

    if not results:
        print("No predictions matched!")
        return pd.DataFrame(columns=["id", "pred_answer"])

    df = pd.DataFrame(results)

    # Deduplicate: keep latest span per dataset row
    df = df.sort_values("start_time").groupby("id").last().reset_index()
    print(f"Matched {len(df)} unique predictions")

    return df[["id", "pred_answer"]]


def _score_row(row):
    """Score a single merged DataFrame row. Returns ScoredExample or None."""
    metric_fn = get_metric(row["secondary_task"])
    if metric_fn is None:
        return None

    gold = parse_answer_list(row["answer"])
    pred = parse_answer_list(row["pred_answer"])
    score = metric_fn(gold, pred)
    if math.isnan(score):
        return None

    return ScoredExample(
        id=row["id"],
        task=get_task_code(row["secondary_task"]),
        metric=metric_fn.__name__,
        score=score,
        gold_answer=gold,
        pred_answer=pred,
        token_length=row.get("token_length"),
        difficulty=row.get("difficulty"),
    )


def _aggregate(results, skipped):
    """Build EvalSummary from a list of ScoredExamples."""
    task_scores = defaultdict(list)
    metric_scores = defaultdict(list)
    length_scores = defaultdict(list)
    difficulty_scores = defaultdict(list)

    for r in results:
        task_scores[r.task].append(r.score)
        metric_scores[r.metric].append(r.score)
        if r.token_length is not None:
            length_scores[r.token_length].append(r.score)
        if r.difficulty is not None:
            difficulty_scores[r.difficulty].append(r.score)

    def _gs(scores):
        return GroupStats(avg=sum(scores) / len(scores) if scores else 0.0, n=len(scores))

    all_scores = [r.score for r in results]
    return EvalSummary(
        per_task={k: _gs(v) for k, v in sorted(task_scores.items())},
        per_metric={k: _gs(v) for k, v in sorted(metric_scores.items())},
        per_length={k: _gs(v) for k, v in length_scores.items()} if length_scores else None,
        per_difficulty={k: _gs(v) for k, v in difficulty_scores.items()} if difficulty_scores else None,
        overall=_gs(all_scores),
        skipped=skipped,
    )


def _print_summary(summary, results, show_errors):
    """Print evaluation summary to stdout."""
    if show_errors:
        for r in results:
            if r.score < 1.0:
                print(f"  [{r.id[:12]}] {r.task} {r.metric}={r.score:.3f}")
                print(f"       gold: {r.gold_answer}")
                print(f"       pred: {r.pred_answer}")
                print()

    print("=" * 60)
    print("Per-task scores:")
    print("-" * 60)
    for code, stats in summary.per_task.items():
        metric_name = get_metric(code).__name__
        print(f"  {code:8s} ({metric_name:20s}): {stats.avg:.3f}  (n={stats.n})")

    print()
    print("Per-metric scores:")
    print("-" * 60)
    for name, stats in summary.per_metric.items():
        print(f"  {name:20s}: {stats.avg:.3f}  (n={stats.n})")

    if summary.per_length:
        print()
        print("Per-context-length scores:")
        print("-" * 60)
        for length in ["8k", "16k", "32k", "64k", "128k", "256k"]:
            if length in summary.per_length:
                stats = summary.per_length[length]
                print(f"  {length:8s}: {stats.avg:.3f}  (n={stats.n})")

    if summary.per_difficulty:
        print()
        print("Per-difficulty scores:")
        print("-" * 60)
        for diff in ["Easy", "Moderate", "Hard", "Extreme"]:
            if diff in summary.per_difficulty:
                stats = summary.per_difficulty[diff]
                print(f"  {diff:10s}: {stats.avg:.3f}  (n={stats.n})")

    print()
    if summary.overall.n:
        print(f"Overall: {summary.overall.avg:.3f}  (n={summary.overall.n}, skipped={summary.skipped})")
    else:
        print(f"No scores computed (skipped={summary.skipped})")
    print("=" * 60)


def _score_predictions(df_gold, df_pred, show_errors=False, output_metrics=None):
    """Score predictions against gold answers. Shared by evaluate and phoenix."""
    merged = df_gold.merge(df_pred[["id", "pred_answer"]], on="id", how="inner")
    print(f"Matched {len(merged)}/{len(df_gold)} examples")

    results = []
    skipped = 0
    for _, row in merged.iterrows():
        r = _score_row(row)
        if r is None:
            skipped += 1
        else:
            results.append(r)

    summary = _aggregate(results, skipped)
    _print_summary(summary, results, show_errors)

    if output_metrics:
        report = EvalReport(summary=summary, examples=results)
        with open(output_metrics, "w") as f:
            json.dump(report.model_dump(), f, indent=2)
        print(f"Metrics written to {output_metrics}")


def evaluate_phoenix(config_path, show_errors=False, limit=10000, output_metrics=None):
    """Evaluate RLM predictions by fetching traces.

    Reads traces_backend, traces_endpoint, traces_project, and dataset.path
    from config YAML. Fetches up to `limit` spans from the project.
    """
    from tracing_backend import make_tracing_backend

    cfg = load_config(config_path)
    backend = make_tracing_backend(cfg.traces_backend or "phoenix", cfg.traces_endpoint)

    df_pred = fetch_predictions(backend, cfg.traces_project, cfg.dataset.path, limit=limit)
    if df_pred.empty:
        return

    df_gold = pd.read_parquet(cfg.dataset.path)
    _score_predictions(df_gold, df_pred, show_errors=show_errors, output_metrics=output_metrics)


if __name__ == "__main__":
    fire.Fire({"evaluate": evaluate, "phoenix": evaluate_phoenix})
