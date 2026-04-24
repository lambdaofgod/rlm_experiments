import pandas as pd
from datasets import Dataset, load_dataset
from unsloth.chat_templates import standardize_data_formats


def load_jsonl_dataset(data_path: str) -> Dataset:
    """Load a JSONL file with 'messages' field into an HF Dataset."""
    return load_dataset("json", data_files=data_path, split="train")


def format_for_sft(dataset: Dataset, tokenizer) -> Dataset:
    """Standardize and apply chat template, producing a 'text' column for SFTTrainer.

    Based on Q3nb cells 13, 17.
    """
    dataset = standardize_data_formats(dataset)

    # standardize_data_formats expects "conversations" key;
    # if our JSONL used "messages", it should already be renamed,
    # but fall back to manual rename if needed
    if "conversations" not in dataset.column_names and "messages" in dataset.column_names:
        dataset = dataset.rename_column("messages", "conversations")

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in convos
        ]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)
    return dataset


def print_token_length_stats(dataset: Dataset, tokenizer) -> None:
    tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    lengths = [len(tok.encode(t)) for t in dataset["text"]]
    print(pd.Series(lengths, name="token_length").describe())
