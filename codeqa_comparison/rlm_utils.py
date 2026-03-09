"""RLM utilities: OpenTelemetry tracing setup."""

from typing import Optional


def setup_otel(endpoint: str, project_name: Optional[str] = None) -> None:
    """Configure OpenTelemetry tracing and instrument RLM.

    Requires the ``rlm[otel]`` extra and ``opentelemetry-exporter-otlp``.
    """
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from rlm.tracing.otel import RLMInstrumentor
    except ImportError as e:
        raise ImportError(
            f"Missing OpenTelemetry dependency: {e}. "
            "Install with: uv add 'rlm[otel]' opentelemetry-exporter-otlp"
        ) from e
    if project_name:
        provider = TracerProvider(resource=Resource({"service.name": project_name}))
    else:
        provider = TracerProvider()
    provider.add_span_processor(
        SimpleSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
    )
    RLMInstrumentor().instrument(tracer_provider=provider)
