"""Utilities for preparing datasets for SFT data collection."""

import json

import fire
from datasets import load_dataset as hf_load_dataset


RLM_BRIDGE = (
    "CRITICAL: we are working in RLM regime, which means instead of "
    "using previous formatting guidance you should output your final "
    "answer via the SUBMIT tool. Call SUBMIT(answer=[...]) with your "
    "answer as a list of values, e.g. "
    'SUBMIT(answer=["val1", "val2"]).'
)


def transform_question(text):
    """Append the RLM bridge sentence; leave the rest of the prompt untouched."""
    return text.rstrip() + "\n\n" + RLM_BRIDGE


def prepare_longbench_pro(output_path="longbench_pro_en.parquet"):
    """Download LongBench-Pro, filter to English, save as parquet.

    Answers (lists) are serialized as JSON strings.
    """
    ds = hf_load_dataset("caskcsg/LongBench-Pro", split="test")
    en = ds.filter(lambda x: x["language"] == "English")
    df = en.to_pandas()
    df["answer"] = df["answer"].apply(lambda x: json.dumps(x.tolist() if hasattr(x, "tolist") else x))
    df["question_thinking"] = df["question_thinking"].apply(transform_question)
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(df)} English rows to {output_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"Token length distribution: {dict(df['token_length'].value_counts())}")


def sample_longbench_pro(input_path="longbench_pro_en.parquet", output_path="sample_longbench_pro_en.parquet"):
    """Sample 1 row per token_length bucket from the prepared parquet."""
    import pandas as pd

    df = pd.read_parquet(input_path)
    sample = df.groupby("token_length", sort=False).first().reset_index()
    sample.to_parquet(output_path, index=False)
    print(f"Saved {len(sample)} rows to {output_path}")
    for _, row in sample.iterrows():
        print(f"  {row['token_length']}: {row['id'][:16]}...")


def build_question_lookup(dataset_df):
    """Map post-transform question_thinking text -> DataFrame row."""
    return {row["question_thinking"]: row for _, row in dataset_df.iterrows()}


def match_question_to_row(query, question_to_row):
    """Match a query string to a dataset row via question_thinking text."""
    return question_to_row.get(query)


if __name__ == "__main__":
    fire.Fire({
        "longbench_pro": prepare_longbench_pro,
        "sample_longbench_pro": sample_longbench_pro,
    })
