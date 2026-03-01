"""DSPy RLM wrapper -- kept for easy switching back to dspy.RLM."""

import dspy
import logging
from typing import Optional

logging.getLogger("dspy").setLevel(logging.DEBUG)


def setup_otel(endpoint: str):
    from openinference.instrumentation.dspy import DSPyInstrumentor
    from opentelemetry import trace as trace_api
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk import trace as trace_sdk
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    tracer_provider = trace_sdk.TracerProvider()
    trace_api.set_tracer_provider(tracer_provider)
    exporter = OTLPSpanExporter(endpoint=endpoint)
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    DSPyInstrumentor().instrument()


def make_rlm(
    model: str,
    base_url: Optional[str] = None,
    otel_endpoint: Optional[str] = None,
):
    """Configure DSPy and return a callable RLM module.

    The returned object is called as ``rlm(context=..., query=...)``
    and returns an object with an ``.answer`` attribute.
    """
    if otel_endpoint is not None:
        setup_otel(otel_endpoint)

    lm_kwargs = {}
    if base_url is not None:
        lm_kwargs["api_base"] = base_url
    dspy.configure(lm=dspy.LM(model, **lm_kwargs))
    return dspy.RLM("context, query -> answer")
