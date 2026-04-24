from pathlib import Path

import yaml
from pydantic import BaseModel, model_validator


class ModelConfig(BaseModel):
    name: str = "unsloth/Qwen3.5-4B"
    max_seq_length: int
    load_in_4bit: bool


class LoraConfig(BaseModel):
    rank: int
    alpha: int
    dropout: float = 0


class TrainingConfig(BaseModel):
    max_steps: int | None = None
    num_train_epochs: int | None = None

    @model_validator(mode="after")
    def check_steps_or_epochs(self) -> "TrainingConfig":
        if self.max_steps is None and self.num_train_epochs is None:
            raise ValueError("Either max_steps or num_train_epochs must be set")
        if self.max_steps is not None and self.num_train_epochs is not None:
            raise ValueError("Set max_steps or num_train_epochs, not both")
        return self
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_steps: int = 5
    weight_decay: float = 0.001
    lr_scheduler_type: str = "linear"
    optim: str = "adamw_8bit"
    seed: int = 3407


class SFTTrainingConfig(BaseModel):
    data_path: str
    output_dir: str
    trackio_project: str | None = None
    gguf_quantizations: list[str] = []
    model: ModelConfig
    lora: LoraConfig
    training: TrainingConfig

    @classmethod
    def load(cls, path: str | Path = "config.yaml") -> "SFTTrainingConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
