"""Export RLM traces as SFT training data in OpenAI chat format.

Each RLM iteration (LM.__call__ span) becomes one JSONL record with
{"messages": [system, user, assistant]}. Optionally filters by eval
score from a --output_metrics JSON file.
"""

import json

import fire
import pandas as pd

from config_model import load_config
from tracing_backend import (
    INPUT_VALUE,
    NAME,
    OUTPUT_VALUE,
    PARENT_ID,
    START_TIME,
    STATUS_CODE,
    TRACE_ID,
    make_tracing_backend,
)


def _build_dataset_lookup(dataset_path):
    """Build query-text -> row lookup tables for matching spans to dataset rows."""
    from data_utils import transform_question

    dataset = pd.read_parquet(dataset_path)

    question_to_row = {}
    for _, row in dataset.iterrows():
        question_to_row[row["question_nonthinking"]] = row

    transformed_to_row = {}
    for _, row in dataset.iterrows():
        transformed = transform_question(row["question_nonthinking"])
        transformed_to_row[transformed] = row

    return question_to_row, transformed_to_row


def _match_trace_to_id(root_span, question_to_row, transformed_to_row):
    """Match an RLM.forward root span to a dataset row ID via query text.

    Reuses the same matching logic as fetch_predictions in evaluate.py.
    Returns (id, question_nonthinking) or (None, None) if no match.
    """
    from data_utils import transform_question

    input_data = json.loads(root_span[INPUT_VALUE])
    query = input_data["input_args"]["query"]

    matched_row = question_to_row.get(query)

    if matched_row is None:
        transformed_query = transform_question(query)
        matched_row = question_to_row.get(transformed_query)

    if matched_row is None:
        matched_row = transformed_to_row.get(query)

    if matched_row is None:
        return None, None

    return matched_row["id"], matched_row["question_nonthinking"]


def _trace_to_training_examples(trace_spans):
    """Extract training examples from all spans in one RLM trace.

    Finds LM.__call__ spans whose parent is a ChatAdapter.__call__ span
    (skips JSONAdapter retries). Each becomes one {"messages": [...]} record.

    Returns list of dicts ready for JSONL.
    """
    # Build parent lookup: span_id -> span name
    parent_names = {}
    for span_id, span in trace_spans.iterrows():
        parent_names[span_id] = span[NAME]

    lm_spans = trace_spans[trace_spans[NAME] == "LM.__call__"].copy()
    lm_spans = lm_spans.sort_values(START_TIME)

    examples = []
    for _, lm_span in lm_spans.iterrows():
        # Skip JSONAdapter retries
        parent_id = lm_span[PARENT_ID]
        parent_name = parent_names.get(parent_id, "")
        if parent_name != "ChatAdapter.__call__":
            continue

        input_data = json.loads(lm_span[INPUT_VALUE])
        output_data = json.loads(lm_span[OUTPUT_VALUE])

        # output_data[0] can be a dict with "text" key or a plain string
        first_output = output_data[0]
        assistant_content = first_output["text"] if isinstance(first_output, dict) else first_output

        messages = []
        for msg in input_data["messages"]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "assistant", "content": assistant_content})

        examples.append({"messages": messages})

    return examples


def _load_passing_ids(metrics_file, min_score):
    """Load dataset IDs that meet the score threshold from a metrics JSON file."""
    with open(metrics_file) as f:
        report = json.load(f)
    passing = set()
    for ex in report["examples"]:
        if ex["score"] >= min_score:
            passing.add(ex["id"])
    return passing


def export_traces(config_path, output="traces.jsonl", metrics_file=None, min_score=1.0, limit=10000):
    """Export RLM traces as SFT training JSONL.

    Args:
        config_path: YAML config with traces_endpoint, traces_project, dataset.path
        output: Output JSONL file path
        metrics_file: Optional metrics JSON from --output_metrics for score filtering
        min_score: Minimum score threshold when using metrics_file (default: 1.0)
        limit: Max spans to fetch
    """
    cfg = load_config(config_path)

    if not cfg.traces_project:
        print("Error: config must have traces_project")
        return

    # Load score filter if provided
    passing_ids = None
    if metrics_file:
        passing_ids = _load_passing_ids(metrics_file, min_score)
        print(f"Loaded {len(passing_ids)} IDs with score >= {min_score} from {metrics_file}")

    # Fetch all spans
    backend = make_tracing_backend(cfg.traces_backend or "phoenix", cfg.traces_endpoint)
    all_spans = backend.get_all_spans(cfg.traces_project, limit=limit)
    print(f"Fetched {len(all_spans)} total spans")

    # Find OK module forward spans
    forward_name = f"{cfg.module.type}.forward"
    root_spans = all_spans[
        (all_spans[NAME] == forward_name) & (all_spans[STATUS_CODE] == "OK")
    ]
    print(f"Found {len(root_spans)} OK {forward_name} traces")

    # Deduplicate: keep latest trace per trace_id
    root_spans = root_spans.sort_values(START_TIME).copy()

    # Build dataset lookups
    question_to_row, transformed_to_row = _build_dataset_lookup(cfg.dataset.path)

    # Process each trace
    all_examples = []
    matched = 0
    unmatched = 0
    filtered_by_score = 0
    seen_ids = {}  # id -> start_time, for deduplication

    for _, root_span in root_spans.iterrows():
        row_id, _ = _match_trace_to_id(root_span, question_to_row, transformed_to_row)

        if row_id is None:
            unmatched += 1
            continue

        # Deduplicate: keep latest trace per dataset row
        span_time = root_span[START_TIME]
        if row_id in seen_ids and seen_ids[row_id] >= span_time:
            continue
        seen_ids[row_id] = span_time

        # Filter by score if metrics file provided
        if passing_ids is not None and row_id not in passing_ids:
            filtered_by_score += 1
            continue

        # Get all spans in this trace
        trace_id = root_span[TRACE_ID]
        trace_spans = all_spans[all_spans[TRACE_ID] == trace_id]

        examples = _trace_to_training_examples(trace_spans)
        all_examples.append((row_id, examples))
        matched += 1

    # For deduplicated rows that were superseded, remove old entries
    # (we process in start_time order, so later ones overwrite earlier)
    final_examples = {row_id: examples for row_id, examples in all_examples}

    total_records = sum(len(exs) for exs in final_examples.values())

    print(f"Matched {matched} traces to dataset rows")
    if unmatched:
        print(f"Warning: {unmatched} traces could not be matched")
    if filtered_by_score:
        print(f"Filtered out {filtered_by_score} traces below score threshold")

    # Write JSONL
    with open(output, "w") as f:
        for examples in final_examples.values():
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

    print(f"Wrote {total_records} training examples from {len(final_examples)} traces to {output}")


if __name__ == "__main__":
    fire.Fire(export_traces)
