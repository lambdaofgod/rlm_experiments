"""SFT training script for Qwen 3.5 (4B).

Based on Qwen 3 notebook (text SFT approach) with Qwen 3.5 model setup.
Uses local JSONL data via data_utils.
"""

from datetime import datetime
from pathlib import Path

import fire
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer, SFTConfig

from config import SFTTrainingConfig
from data_utils import load_jsonl_dataset, format_for_sft, print_token_length_stats


def main(config_path: str = "config.yaml"):
    config = SFTTrainingConfig.load(config_path)

    # --- Model loading (Q3.5nb cell-6) ---
    model, tokenizer = FastLanguageModel.from_pretrained(
        config.model.name,
        max_seq_length=config.model.max_seq_length,
        load_in_4bit=config.model.load_in_4bit,
        use_gradient_checkpointing="unsloth",
    )

    # --- LoRA (Q3.5nb cell-8) ---
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        bias="none",
        random_state=config.training.seed,
        use_rslora=False,
        loftq_config=None,
    )

    # --- Chat template (Q3nb cell-10) ---
    tokenizer = get_chat_template(tokenizer, chat_template="qwen3-instruct")

    # --- Load and format data ---
    dataset = load_jsonl_dataset(config.data_path)
    dataset = format_for_sft(dataset, tokenizer)
    print_token_length_stats(dataset, tokenizer)

    # --- SFTTrainer (Q3nb cell-21) ---
    t = config.training
    sft_config = SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=t.batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        warmup_steps=t.warmup_steps,
        max_steps=t.max_steps if t.max_steps is not None else -1,
        num_train_epochs=t.num_train_epochs if t.num_train_epochs is not None else 1,
        learning_rate=t.learning_rate,
        logging_steps=1,
        optim=t.optim,
        weight_decay=t.weight_decay,
        lr_scheduler_type=t.lr_scheduler_type,
        seed=t.seed,
        report_to="trackio" if config.trackio_project else "none",
        project=config.trackio_project or "huggingface",
        run_name=f"{config.trackio_project}_{datetime.now():%Y%m%d_%H%M%S}" if config.trackio_project else None,
        trackio_space_id=None,
        output_dir=config.output_dir,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=sft_config,
    )

    # --- Train on responses only (Q3nb cell-23) ---
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    # --- Train (Q3nb cells 28-31) ---
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    print(f"{trainer_stats.metrics['train_runtime']:.1f} seconds used for training.")
    print(f"{trainer_stats.metrics['train_runtime'] / 60:.2f} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {round(used_memory / max_memory * 100, 3)} %.")

    # --- Save LoRA adapters (Q3nb cell-35) ---
    lora_dir = str(Path(config.output_dir) / "lora")
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    print(f"LoRA adapters saved to {lora_dir}")

    # --- GGUF export ---
    for quant in config.gguf_quantizations:
        gguf_dir = str(Path(config.output_dir) / f"gguf_{quant}")
        try:
            model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method=quant)
            print(f"GGUF model saved to {gguf_dir} (quantization: {quant})")
        except Exception as e:
            print(f"GGUF export failed for quantization '{quant}': {e}")


if __name__ == "__main__":
    fire.Fire(main)
