import json
import os

import dspy
import fire
import pandas as pd
import yaml
from dspy.teleprompt import BootstrapFinetune


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


def load_dataset(path, label_field, prompt_template):
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

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


def setup_tracing(endpoint, project_name=None):
    from openinference.instrumentation.dspy import DSPyInstrumentor
    from phoenix.otel import register

    tracer_provider = register(endpoint=endpoint, project_name=project_name)
    DSPyInstrumentor().instrument(tracer_provider=tracer_provider)
    print(f"OTEL tracing enabled -> {endpoint} (project: {project_name})")


def main(config_path="config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    traces_endpoint = cfg.get("traces_endpoint")
    if traces_endpoint:
        setup_tracing(traces_endpoint, project_name=cfg.get("traces_project"))

    lm = dspy.LM(cfg["lm"]["model"], max_tokens=cfg["lm"]["max_tokens"])
    dspy.configure(lm=lm)

    trainset = load_dataset(
        cfg["dataset"]["path"],
        cfg["dataset"]["label_field"],
        cfg["dataset"]["prompt_template"],
    )

    module_args = (
        cfg["module"]["type"],
        cfg["module"].get("signature", "context, query -> answer"),
        cfg["module"].get("kwargs", {}),
    )
    student = build_program(*module_args)
    student.set_lm(lm)
    teacher = build_program(*module_args)

    metrics = {"exact_match": exact_match_metric, "always_true": always_true_metric}
    metric_name = cfg["collection"].get("metric", "exact_match")
    metric = metrics[metric_name]

    collector = SFTDataCollector(
        output_dir=cfg["collection"]["output_dir"],
        metric=metric,
        num_threads=cfg["collection"]["num_threads"],
    )
    collector.compile(student=student, trainset=trainset, teacher=teacher)
    print(f"SFT data written to {cfg['collection']['output_dir']}")


if __name__ == "__main__":
    fire.Fire(main)
