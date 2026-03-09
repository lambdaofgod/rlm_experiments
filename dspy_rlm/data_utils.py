"""Utilities for preparing datasets for SFT data collection."""

import json
import re

import fire
from datasets import load_dataset as hf_load_dataset


def transform_question(text):
    """Strip LongBench-Pro [Answer] format instructions and add SUBMIT hint."""
    # Remove "Output example:" block and everything after it
    # Handle case variations and full-width colon
    text = re.split(r"\n*Output [Ee]xample[:\uff1a]", text, maxsplit=1)[0]

    # Remove the "Output the "[Answer]" identifier first..." sentence
    # Variations: missing spaces, different endings ("without any additional content",
    # "without any other", "one per line", etc.)
    text = re.sub(
        r"""Output the\s*"\[Answer\]"\s*identifier\s*(?:first)?[^.]*\.""",
        "",
        text,
        flags=re.DOTALL,
    )

    text = text.rstrip()
    text += "\n\nReturn your answer as a list of values, e.g. SUBMIT(answer=[\"val1\", \"val2\"])."
    return text


def prepare_longbench_pro(output_path="longbench_pro_en.parquet"):
    """Download LongBench-Pro, filter to English, save as parquet.

    Answers (lists) are serialized as JSON strings.
    """
    ds = hf_load_dataset("caskcsg/LongBench-Pro", split="test")
    en = ds.filter(lambda x: x["language"] == "English")
    df = en.to_pandas()
    df["answer"] = df["answer"].apply(lambda x: json.dumps(x.tolist() if hasattr(x, "tolist") else x))
    df["question_nonthinking"] = df["question_nonthinking"].apply(transform_question)
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


if __name__ == "__main__":
    fire.Fire({
        "longbench_pro": prepare_longbench_pro,
        "sample_longbench_pro": sample_longbench_pro,
    })
