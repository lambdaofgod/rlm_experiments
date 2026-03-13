# DSPy RLM SFT Data Collection & Evaluation

Collect SFT training data from DSPy RLM (Recursive Language Model) runs,
evaluate predictions against LongBench-Pro gold answers, and export
traces from Phoenix for finetuning.

## Setup

```
uv sync
```

Requires a tracing backend (Phoenix or MLflow) for trace-based workflows.

## Config

All scripts share a YAML config file. See `config.yaml` for an example.

```yaml
traces_endpoint: "http://localhost:6006/v1/traces"
traces_project: "dspy-sft-longbench"

lm:
  model: "openrouter/qwen/qwen3.5-122B-A10B"
  max_tokens: 32000
  api_base: "http://localhost:8080/v1"  # optional, for OpenAI-compatible endpoints
  api_key: "dummy"                      # optional, for endpoints requiring auth
  timeout: 60                           # optional, per-request timeout in seconds

dataset:
  path: "longbench_pro_en.parquet"
  label_field: "answer"
  prompt_template:
    context: "{context}"
    query: "{question_nonthinking}"
  token_lengths: ["8k", "16k", "32k"]    # optional filter
  difficulties: ["Easy", "Moderate"]      # optional filter

module:
  type: "CustomizableRLM"
  signature: "context, query -> answer"
  kwargs: {}

collection:
  num_threads: 1
  metric: "always_true"    # or "exact_match"
  output_dir: "./sft_data"
```

Config is validated with Pydantic -- typos in keys will raise errors.

## Scripts

### collect_sft_data.py -- Run RLM and collect SFT data

Runs the DSPy program (teacher LM) over the dataset and writes SFT
training data as JSONL via `BootstrapFinetune`.

```
uv run python collect_sft_data.py --config_path=config.yaml
```

Output goes to `collection.output_dir` (default `./sft_data/`).
Traces are sent to Phoenix if `traces_endpoint` is set.

### evaluate.py -- Evaluate predictions

Two modes: from traces (via `--config_path`) or from a predictions file
(via `--predictions_path`).

**From traces (Phoenix or MLflow):**

```
uv run python evaluate.py \
  --config_path=config.yaml \
  --show_errors \
  --output_metrics=metrics.json
```

**From a predictions file:**

```
uv run python evaluate.py \
  --predictions_path=predictions.csv \
  --dataset_path=longbench_pro_en.parquet \
  --show_errors \
  --output_metrics=metrics.json
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config_path` | `None` | YAML config for trace-based eval (`traces_endpoint`, `traces_project`, `module.type`) |
| `--predictions_path` | `None` | CSV/parquet with `id` and `pred_answer` columns |
| `--dataset_path` | `longbench_pro_en.parquet` | Gold answers parquet (used with `--predictions_path`) |
| `--pred_col` | `pred_answer` | Column name for predictions |
| `--id_col` | `id` | Column name for row IDs |
| `--show_errors` | `False` | Print mismatched predictions |
| `--limit` | `10000` | Max spans to fetch |
| `--output_metrics` | `None` | Write detailed JSON report to this path |

Scores are stratified by task type, metric, context length, and difficulty.

### export_traces.py -- Export traces as SFT training data

Extracts per-iteration LM calls from Phoenix RLM traces and writes
them as OpenAI chat format JSONL -- ready for finetuning.

```
uv run python export_traces.py \
  --config_path=config.yaml \
  --output=traces.jsonl
```

With score filtering (only export traces that scored well):

```
uv run python export_traces.py \
  --config_path=config.yaml \
  --output=traces_filtered.jsonl \
  --metrics_file=metrics.json \
  --min_score=1.0
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config_path` | (required) | YAML config with `traces_endpoint`, `traces_project` |
| `--output` | `traces.jsonl` | Output JSONL path |
| `--metrics_file` | `None` | Metrics JSON from `--output_metrics` for filtering |
| `--min_score` | `1.0` | Minimum score threshold (used with `--metrics_file`) |
| `--limit` | `10000` | Max spans to fetch |

Each output record is one RLM iteration:

```json
{"messages": [
  {"role": "system", "content": "<DSPy signature instructions>"},
  {"role": "user", "content": "<variables_info + repl_history + iteration>"},
  {"role": "assistant", "content": "<reasoning + code>"}
]}
```

### compare_evals.py -- Compare evaluation results across models

Compares two or more evaluation results side by side. Uses an outer
join on problem IDs so that failures are visible -- missing predictions
count as score 0 in the adjusted metrics.

```
uv run python compare_evals.py \
  --dataset=longbench_pro_en.parquet \
  easy_medium_metrics.json local_metrics.json
```

Reports:
- **Completion matrix** -- which problems each model scored vs missed
- **Head-to-head** -- wins/ties/losses on shared problems, per-task breakdown
- **Adjusted overall scores** -- sum(scores) / total_dataset_size
- **Per-dimension breakdowns** -- by token_length and difficulty

## Typical workflow

```
# 1. Run RLM over dataset, traces go to Phoenix
uv run python collect_sft_data.py --config_path=config.yaml

# 2. Evaluate predictions from traces
uv run python evaluate.py \
  --config_path=config.yaml \
  --output_metrics=metrics.json

# 3. Export only high-scoring traces for finetuning
uv run python export_traces.py \
  --config_path=config.yaml \
  --metrics_file=metrics.json \
  --min_score=1.0 \
  --output=sft_train.jsonl

# 4. Compare results between models
uv run python compare_evals.py \
  --dataset=longbench_pro_en.parquet \
  metrics_model_a.json metrics_model_b.json
```
