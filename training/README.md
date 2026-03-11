# training

SFT fine-tuning pipeline for Qwen 3.5 (4B) using Unsloth and TRL, with YAML-driven configuration and GGUF export.

## Overview

- **config.py** - Pydantic config schema (model, LoRA, training params) loaded from YAML
- **data_utils.py** - JSONL data loading and chat-template formatting for SFT
- **train_qwen3_5.py** - Main training script: loads model, applies LoRA, runs SFT, saves adapters
- **export_gguf.py** - Merges LoRA adapters into base model and exports as GGUF
- **config.yaml** - Default training configuration

## Setup

```bash
uv sync
```

Requires a CUDA GPU with sufficient VRAM for 4-bit Qwen 3.5 (4B).

## Usage

### Training

```bash
uv run python train_qwen3_5.py                    # uses config.yaml
uv run python train_qwen3_5.py --config_path my_config.yaml
```

Training data should be a JSONL file where each line has a `messages` field (OpenAI chat format). The script:

1. Loads the model in 4-bit with Unsloth
2. Applies LoRA adapters
3. Formats data using the Qwen3 instruct chat template
4. Trains on assistant responses only
5. Saves LoRA adapters to `<output_dir>/lora/`

### GGUF Export

```bash
uv run python export_gguf.py --config_path config.yaml
uv run python export_gguf.py --config_path config.yaml --quantization q4_k_m
```

Loads the LoRA adapters from `<output_dir>/lora/`, merges them into the base model, and exports to `<output_dir>/gguf/`.

## Configuration

All parameters are set in `config.yaml`:

| Section    | Key                          | Description                          |
|------------|------------------------------|--------------------------------------|
| (root)     | `data_path`                  | Path to training JSONL               |
| (root)     | `output_dir`                 | Output directory for checkpoints     |
| (root)     | `trackio_project`            | TrackIO project name (null to skip)  |
| `model`    | `name`                       | HF model ID                          |
| `model`    | `max_seq_length`             | Max sequence length                  |
| `model`    | `load_in_4bit`               | 4-bit quantization                   |
| `lora`     | `rank`, `alpha`, `dropout`   | LoRA hyperparameters                 |
| `training` | `num_train_epochs`/`max_steps` | Exactly one must be set            |
| `training` | `batch_size`, `learning_rate`, `warmup_steps`, etc. | Standard training args |
