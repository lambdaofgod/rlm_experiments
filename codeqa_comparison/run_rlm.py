"""Run RLM on the CodeQA (Code Repository Understanding) subset of LongBench-v2."""

import json
import pathlib
from datetime import datetime, timezone
from typing import Optional

import fire
import pandas as pd
from datasets import load_dataset
from rlm import RLM
from rlm.logger import RLMLogger
from rlm.utils.exceptions import TokenLimitExceededError
from rlm.utils.token_utils import MODEL_CONTEXT_LIMITS
from tqdm import tqdm

from rlm_prompts import get_rlm_system_prompt

CHOICES = ["A", "B", "C", "D"]

QUERY_TEMPLATE = """\
You are given the context of a code repository and a multiple-choice question about it.

{query}"""


def load_codeqa():
    ds = load_dataset("THUDM/LongBench-v2", split="train")
    return ds.filter(lambda ex: ex["domain"] == "Code Repository Understanding")


def load_annotated_codeqa(path: str = "annotated_dataset.parquet"):
    """Load annotated CodeQA dataset with instruction_suffix appended to prompt."""
    df = pd.read_parquet(path)
    df["prompt"] = df["question"] + "\n\n" + df["instruction_suffix"]
    return df


def build_query(example):
    question = example.get("prompt", example["question"])
    lines = [question, ""]
    for ch in CHOICES:
        lines.append(f"{ch}. {example[f'choice_{ch}']}")
    lines.append("")
    lines.append("The final answer should be a single letter: A, B, C, or D.")
    return "\n".join(lines)


def extract_choice(answer_text: str) -> str:
    for ch in CHOICES:
        if ch in answer_text.upper():
            return ch
    return answer_text.strip()[:1].upper()


def run_single(rlm: RLM, example) -> tuple[str, str]:
    query = build_query(example)
    root_prompt = QUERY_TEMPLATE.format(query=query)
    context = example["context"]
    print(
        f"\n{'='*80}\nROOT PROMPT ({len(root_prompt)} chars):\n{'='*80}\n{root_prompt}\n{'='*80}"
    )
    print(f"CONTEXT: {len(context)} chars")
    result = rlm.completion(context, root_prompt=root_prompt)
    raw = result.response
    predicted = extract_choice(raw)
    return predicted, raw


def make_record(example, predicted, raw_answer, model):
    return {
        "id": example["_id"],
        "question": example["question"],
        "gold": example["answer"],
        "predicted": predicted,
        "raw_answer": raw_answer,
        "correct": predicted == example["answer"],
        "sub_domain": example["sub_domain"],
        "difficulty": example["difficulty"],
        "length": example["length"],
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main(
    model: str = "openai/local",
    base_url: Optional[str] = None,
    output: Optional[str] = None,
    dataset: str = "original",
    max_tokens: int = 40960,
    environment: str = "pyodide",
    log_dir: Optional[str] = "rlm_logs",
    verbose: bool = False,
):
    """Run RLM on every CodeQA question and write results to a JSONL file.

    Args:
        model: LiteLLM model string, e.g. "openai/gpt-5" or "openrouter/google/gemini-2.5-pro-preview".
        base_url: Custom OpenAI-compatible API base URL.
        output: Path to the output JSONL file.
        dataset: "original" or "annotated".
        max_tokens: Maximum total tokens (input + output) for RLM.
        environment: RLM REPL environment ("pyodide" or "local").
        log_dir: Directory for RLM JSONL logs. Set to None to disable.
        verbose: Enable RLM verbose console output.
    """
    if output is None:
        output = (
            "results_rlm_annotated.jsonl"
            if dataset == "annotated"
            else "results_rlm.jsonl"
        )

    backend_kwargs = {"model_name": model}
    if base_url is not None:
        backend_kwargs["api_base"] = base_url

    if base_url is not None:
        MODEL_CONTEXT_LIMITS[model] = max_tokens

    logger = RLMLogger(log_dir=log_dir) if log_dir else None

    rlm = RLM(
        backend="litellm",
        backend_kwargs=backend_kwargs,
        environment=environment,
        max_tokens=max_tokens,
        compaction=True,
        custom_system_prompt=get_rlm_system_prompt(max_tokens),
        logger=logger,
        verbose=verbose,
    )

    n_examples = None
    if dataset.endswith(".parquet"):
        df = load_annotated_codeqa(dataset)
        n_examples = len(df)
        ds = (row for _, row in df.iterrows())
    elif dataset == "annotated":
        df = load_annotated_codeqa()
        n_examples = len(df)
        ds = (row for _, row in df.iterrows())
    else:
        ds = load_codeqa()
    out_path = pathlib.Path(output)

    correct = 0
    total = 0

    with out_path.open("w") as f:
        pbar = tqdm(ds, total=n_examples, desc="CodeQA RLM")
        for example in pbar:
            try:
                predicted, raw_answer = run_single(rlm, example)
            except (TokenLimitExceededError, Exception) as e:
                if isinstance(e, TokenLimitExceededError) or "context" in str(e).lower():
                    tqdm.write(f"  SKIPPED {example['_id']}: {e}")
                    continue
                raise
            record = make_record(example, predicted, raw_answer, model)

            correct += record["correct"]
            total += 1

            pbar.set_postfix(
                acc=f"{correct}/{total}", last=f"{record['predicted']}/{record['gold']}"
            )
            f.write(json.dumps(record) + "\n")
            f.flush()

    print(f"\nFinal accuracy: {correct}/{total} ({correct/total:.1%})")


if __name__ == "__main__":
    fire.Fire(main)
