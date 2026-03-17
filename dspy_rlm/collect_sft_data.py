import json
import logging
import traceback

import dspy
import fire
import pandas as pd
from tqdm import tqdm

from config_model import load_config

logger = logging.getLogger(__name__)


def load_dataset(path, label_field, prompt_template, token_lengths=None, difficulties=None):
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if token_lengths and "token_length" in df.columns:
        before = len(df)
        df = df[df["token_length"].isin(token_lengths)]
        logger.info("Filtered to token_lengths %s: %d -> %d rows", token_lengths, before, len(df))

    if difficulties and "difficulty" in df.columns:
        before = len(df)
        df = df[df["difficulty"].isin(difficulties)]
        logger.info("Filtered to difficulties %s: %d -> %d rows", difficulties, before, len(df))

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
    from config_model import load_module_class

    module_cls = load_module_class(module_type)
    return module_cls(signature, **kwargs)


def setup_tracing(backend, endpoint, project_name=None):
    if backend == "mlflow":
        import mlflow

        if endpoint:
            mlflow.set_tracking_uri(endpoint)
        if project_name:
            mlflow.set_experiment(project_name)
        mlflow.dspy.autolog()
        logger.info("MLflow tracing enabled -> %s (experiment: %s)", endpoint or "default", project_name)
        return

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
        logger.warning("Unknown traces_backend: %r, skipping tracing", backend)
        return

    DSPyInstrumentor().instrument(tracer_provider=tracer_provider)
    logger.info("OTEL tracing enabled (%s) -> %s (project: %s)", backend, endpoint, project_name)


def main(config_path="config.yaml"):
    cfg = load_config(config_path)

    if cfg.traces_backend or cfg.traces_endpoint:
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

    program = build_program(cfg.module.type, cfg.module.signature, cfg.module.kwargs)
    program.set_lm(lm)

    metrics = {"exact_match": exact_match_metric, "always_true": always_true_metric}
    metric = metrics[cfg.collection.metric]

    succeeded, failed = 0, 0
    pbar = tqdm(trainset, desc="Collecting traces")
    for example in pbar:
        try:
            prediction = program(**example.inputs())
            score = metric(example, prediction)
            succeeded += 1
            status = "OK" if score else "WRONG"
        except Exception:
            failed += 1
            status = "ERROR"
            traceback.print_exc()
        pbar.set_postfix(ok=succeeded, err=failed, last=status)

    logger.info("Done: %d succeeded, %d failed out of %d", succeeded, failed, len(trainset))


if __name__ == "__main__":
    fire.Fire(main)
