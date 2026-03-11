import json
import os

import dspy
import fire
import pandas as pd
from dspy.teleprompt import BootstrapFinetune

from config_model import load_config


class SFTDataCollector(BootstrapFinetune):
    def __init__(self, output_dir, **kwargs):
        super().__init__(**kwargs)
        self.output_dir = output_dir

    def compile(self, student, trainset, teacher=None):
        lms = {pred.lm or dspy.settings.lm for pred in student.predictors()}
        for lm in lms:
            lm.finetune = self._make_noop_finetune(lm)
        return super().compile(student, trainset=trainset, teacher=teacher)

    def _make_noop_finetune(self, lm):
        collector = self

        def noop_finetune(train_data, **kwargs):
            os.makedirs(collector.output_dir, exist_ok=True)
            path = os.path.join(collector.output_dir, f"{lm.model.replace('/', '_')}.jsonl")
            with open(path, "w") as f:
                for record in train_data:
                    f.write(json.dumps(record) + "\n")
            print(f"Wrote {len(train_data)} records to {path}")
            return type(
                "NoOpJob", (), {
                    "result": lambda self: lm,
                    "thread": type("NoOpThread", (), {"join": lambda self: None})(),
                },
            )()

        return noop_finetune


def load_dataset(path, label_field, prompt_template, token_lengths=None, difficulties=None):
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if token_lengths and "token_length" in df.columns:
        before = len(df)
        df = df[df["token_length"].isin(token_lengths)]
        print(f"Filtered to token_lengths {token_lengths}: {before} -> {len(df)} rows")

    if difficulties and "difficulty" in df.columns:
        before = len(df)
        df = df[df["difficulty"].isin(difficulties)]
        print(f"Filtered to difficulties {difficulties}: {before} -> {len(df)} rows")

    examples = []
    input_fields = list(prompt_template.keys())
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        rendered = {field: tpl.format(**row_dict) for field, tpl in prompt_template.items()}
        rendered["answer"] = str(row_dict[label_field])
        ex = dspy.Example(**rendered).with_inputs(*input_fields)
        examples.append(ex)
    return examples


def always_true_metric(example, prediction, trace=None):
    return True


def exact_match_metric(example, prediction, trace=None):
    gold = str(example.answer).strip().lower()
    pred = str(prediction.answer).strip().lower()
    if gold == pred:
        return True
    # Try parsing as JSON list for datasets with list-valued answers
    try:
        gold_list = json.loads(gold)
        if isinstance(gold_list, list):
            gold_joined = ", ".join(str(x).strip().lower() for x in gold_list)
            return gold_joined == pred
    except (json.JSONDecodeError, TypeError):
        pass
    return False


def build_program(module_type, signature, kwargs):
    module_cls = getattr(dspy, module_type, None)
    if module_cls is None:
        raise ValueError(f"Unknown dspy module type: {module_type}")
    return module_cls(signature, **kwargs)


def setup_tracing(backend, endpoint, project_name=None):
    from openinference.instrumentation.dspy import DSPyInstrumentor

    if backend == "phoenix":
        from phoenix.otel import register

        tracer_provider = register(endpoint=endpoint, project_name=project_name)
    elif backend == "otel":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
        )
    else:
        print(f"Unknown traces_backend: {backend!r}, skipping tracing")
        return

    DSPyInstrumentor().instrument(tracer_provider=tracer_provider)
    print(f"OTEL tracing enabled ({backend}) -> {endpoint} (project: {project_name})")


def main(config_path="config.yaml"):
    cfg = load_config(config_path)

    if cfg.traces_endpoint:
        backend = cfg.traces_backend or "phoenix"
        setup_tracing(backend, cfg.traces_endpoint, project_name=cfg.traces_project)

    lm_kwargs = {}
    if cfg.lm.api_base:
        lm_kwargs["api_base"] = cfg.lm.api_base
    if cfg.lm.api_key:
        lm_kwargs["api_key"] = cfg.lm.api_key
    if cfg.lm.timeout:
        lm_kwargs["timeout"] = cfg.lm.timeout
    lm = dspy.LM(cfg.lm.model, max_tokens=cfg.lm.max_tokens, **lm_kwargs)
    dspy.configure(lm=lm)

    trainset = load_dataset(
        cfg.dataset.path,
        cfg.dataset.label_field,
        cfg.dataset.prompt_template,
        token_lengths=cfg.dataset.token_lengths,
        difficulties=cfg.dataset.difficulties,
    )

    student = build_program(cfg.module.type, cfg.module.signature, cfg.module.kwargs)
    student.set_lm(lm)
    teacher = build_program(cfg.module.type, cfg.module.signature, cfg.module.kwargs)

    metrics = {"exact_match": exact_match_metric, "always_true": always_true_metric}
    metric = metrics[cfg.collection.metric]

    collector = SFTDataCollector(
        output_dir=cfg.collection.output_dir,
        metric=metric,
        num_threads=cfg.collection.num_threads,
    )
    collector.compile(student=student, trainset=trainset, teacher=teacher)
    print(f"SFT data written to {cfg.collection.output_dir}")


if __name__ == "__main__":
    fire.Fire(main)
