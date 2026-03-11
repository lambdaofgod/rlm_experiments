"""Pydantic models for YAML config validation."""

from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict


TokenLength = Literal["8k", "16k", "32k", "64k", "128k", "256k"]
Difficulty = Literal["Easy", "Moderate", "Hard", "Extreme"]
MetricName = Literal["exact_match", "always_true"]


class LMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    max_tokens: int = 32000
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    timeout: Optional[int] = None


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    label_field: str = "answer"
    prompt_template: dict[str, str]
    token_lengths: Optional[list[TokenLength]] = None
    difficulties: Optional[list[Difficulty]] = None


class ModuleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    signature: str = "context, query -> answer"
    kwargs: dict = {}


class CollectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_threads: int = 1
    metric: MetricName = "exact_match"
    output_dir: str = "./sft_data"


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    traces_backend: Optional[str] = None
    traces_endpoint: Optional[str] = None
    traces_project: Optional[str] = None
    lm: LMConfig
    dataset: DatasetConfig
    module: ModuleConfig = ModuleConfig(type="RLM")
    collection: CollectionConfig = CollectionConfig()


# ---------------------------------------------------------------------------
# Eval output models
# ---------------------------------------------------------------------------


class GroupStats(BaseModel):
    avg: float
    n: int


class ScoredExample(BaseModel):
    id: str
    task: str
    metric: str
    score: float
    gold_answer: list[str]
    pred_answer: list[str]
    token_length: Optional[str] = None
    difficulty: Optional[str] = None


class EvalSummary(BaseModel):
    per_task: dict[str, GroupStats]
    per_metric: dict[str, GroupStats]
    per_length: Optional[dict[str, GroupStats]] = None
    per_difficulty: Optional[dict[str, GroupStats]] = None
    overall: GroupStats
    skipped: int


class EvalReport(BaseModel):
    summary: EvalSummary
    examples: list[ScoredExample]


def load_config(config_path: str) -> Config:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)
