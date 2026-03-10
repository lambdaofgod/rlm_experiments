"""Export LoRA fine-tuned model to GGUF format.

Loads base model + LoRA adapters, merges, and exports as GGUF.
Uses unsloth's built-in save_pretrained_gguf (Q3.5nb GGUF section).
"""

from pathlib import Path

import fire
from unsloth import FastLanguageModel

from config import SFTTrainingConfig


def main(
    config_path: str,
    output_dir: str | None = None,
    quantization: str = "q8_0",
):
    config = SFTTrainingConfig.load(config_path)
    lora_dir = str(Path(config.output_dir) / "lora")

    if output_dir is None:
        output_dir = str(Path(config.output_dir) / "gguf")

    model, tokenizer = FastLanguageModel.from_pretrained(
        lora_dir,
        max_seq_length=config.model.max_seq_length,
        load_in_4bit=config.model.load_in_4bit,
    )

    model.save_pretrained_gguf(output_dir, tokenizer, quantization_method=quantization)
    print(f"GGUF model saved to {output_dir} (quantization: {quantization})")


if __name__ == "__main__":
    fire.Fire(main)
