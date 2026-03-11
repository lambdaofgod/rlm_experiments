"""Abstract tracing backend with Phoenix and MLflow implementations.

Canonical span DataFrame schema used by all downstream code:

    span_id        str        Unique span identifier (DataFrame index)
    trace_id       str        Groups spans into a trace
    parent_id      str/None   Parent span ID
    name           str        Span name (e.g. "RLM.forward")
    status_code    str        "OK" or "ERROR"
    status_message str/None   Error message if status=ERROR
    start_time     datetime   Span start
    end_time       datetime   Span end
    input_value    str        JSON string of input args
    output_value   str        JSON string of output
"""

import re
from abc import ABC, abstractmethod

import pandas as pd


# Canonical column names
SPAN_ID = "span_id"
TRACE_ID = "trace_id"
PARENT_ID = "parent_id"
NAME = "name"
STATUS_CODE = "status_code"
STATUS_MESSAGE = "status_message"
START_TIME = "start_time"
END_TIME = "end_time"
INPUT_VALUE = "input_value"
OUTPUT_VALUE = "output_value"


class TracingBackend(ABC):
    @abstractmethod
    def get_root_spans(self, project_name: str, limit: int = 10000) -> pd.DataFrame:
        """Fetch root-level spans for a project.

        Returns a DataFrame with the canonical schema, indexed by span_id.
        """
        ...

    @abstractmethod
    def get_all_spans(self, project_name: str, limit: int = 10000) -> pd.DataFrame:
        """Fetch all spans (including children) for a project.

        Returns a DataFrame with the canonical schema, indexed by span_id.
        """
        ...


# Phoenix column name -> canonical column name
_PHOENIX_COLUMN_MAP = {
    "context.trace_id": TRACE_ID,
    "parent_id": PARENT_ID,
    "name": NAME,
    "status_code": STATUS_CODE,
    "status_message": STATUS_MESSAGE,
    "start_time": START_TIME,
    "end_time": END_TIME,
    "attributes.input.value": INPUT_VALUE,
    "attributes.output.value": OUTPUT_VALUE,
}


class PhoenixBackend(TracingBackend):
    def __init__(self, endpoint: str):
        import httpx
        from phoenix.client import Client

        base_url = re.sub(r"/v1/traces/?$", "", endpoint)
        http_client = httpx.Client(
            base_url=base_url,
            timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0),
        )
        self._client = Client(http_client=http_client)

    def get_root_spans(self, project_name: str, limit: int = 10000) -> pd.DataFrame:
        df = self._client.spans.get_spans_dataframe(
            project_name=project_name,
            root_spans_only=True,
            limit=limit,
            timeout=300,
        )
        return self._normalize(df)

    def get_all_spans(self, project_name: str, limit: int = 10000) -> pd.DataFrame:
        df = self._client.spans.get_spans_dataframe(
            project_name=project_name,
            root_spans_only=False,
            limit=limit,
            timeout=300,
        )
        return self._normalize(df)

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns=_PHOENIX_COLUMN_MAP)
        df.index.name = SPAN_ID
        return df


class MlflowBackend(TracingBackend):
    """Tracing backend that reads spans from an MLflow tracking server.

    MLflow stores DSPy traces via mlflow.dspy.autolog(). The span data
    format differs from Phoenix/OpenInference:

    - Inputs are raw kwargs dicts (not wrapped in {"input_args": ...})
    - Outputs for module spans are dicts (not Prediction repr strings)

    This backend normalizes both to match the canonical schema expected
    by downstream code (evaluate.py, export_traces.py).
    """

    def __init__(self, endpoint: str | None = None):
        import mlflow

        if endpoint:
            mlflow.set_tracking_uri(endpoint)
        self._client = mlflow.client.MlflowClient()

    def _get_experiment_id(self, experiment_name: str) -> str:
        experiment = self._client.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(
                f"MLflow experiment {experiment_name!r} not found. "
                f"Create it first or check the experiment name."
            )
        return experiment.experiment_id

    def get_root_spans(self, project_name: str, limit: int = 10000) -> pd.DataFrame:
        traces = self._client.search_traces(
            locations=[self._get_experiment_id(project_name)],
            max_results=limit,
        )
        rows = []
        for trace in traces:
            if not trace.data.spans:
                continue
            root = trace.data.spans[0]
            rows.append(self._span_to_row(root, trace.info.trace_id))
        return self._to_dataframe(rows)

    def get_all_spans(self, project_name: str, limit: int = 10000) -> pd.DataFrame:
        traces = self._client.search_traces(
            locations=[self._get_experiment_id(project_name)],
            max_results=limit,
        )
        rows = []
        for trace in traces:
            for span in trace.data.spans:
                rows.append(self._span_to_row(span, trace.info.trace_id))
        return self._to_dataframe(rows)

    @staticmethod
    def _span_to_row(span, trace_id: str) -> dict:
        import json

        inputs = span.inputs or {}
        outputs = span.outputs

        # Normalize inputs to match Phoenix/OpenInference format.
        # For module forward spans (e.g. RLM.forward), Phoenix wraps
        # kwargs in {"input_args": {...}} -- downstream code accesses
        # input_data["input_args"]["query"].  MLflow stores raw kwargs.
        # For LM call spans, Phoenix stores {"messages": [...]} at
        # top level -- downstream code accesses input_data["messages"].
        # Only wrap in input_args when it's not an LM-style span.
        if "input_args" not in inputs and "messages" not in inputs:
            inputs = {"input_args": inputs}

        # Timestamps: MLflow uses nanoseconds since epoch
        start_time = pd.Timestamp(span.start_time_ns, unit="ns")
        end_time = (
            pd.Timestamp(span.end_time_ns, unit="ns")
            if span.end_time_ns
            else pd.NaT
        )

        # Status: span.status is a SpanStatus with .status_code (enum) and .description
        status = span.status
        status_code = status.status_code.name if status else "OK"
        # Treat UNSET as OK (completed without explicit status)
        if status_code == "UNSET":
            status_code = "OK"
        status_message = status.description if status else None

        return {
            SPAN_ID: span.span_id,
            TRACE_ID: trace_id,
            PARENT_ID: span.parent_id,
            NAME: span.name,
            STATUS_CODE: status_code,
            STATUS_MESSAGE: status_message,
            START_TIME: start_time,
            END_TIME: end_time,
            INPUT_VALUE: json.dumps(inputs, default=str),
            OUTPUT_VALUE: json.dumps(outputs, default=str),
        }

    @staticmethod
    def _to_dataframe(rows: list[dict]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(
                columns=[
                    SPAN_ID, TRACE_ID, PARENT_ID, NAME, STATUS_CODE,
                    STATUS_MESSAGE, START_TIME, END_TIME, INPUT_VALUE, OUTPUT_VALUE,
                ]
            ).set_index(SPAN_ID)
        df = pd.DataFrame(rows)
        df = df.set_index(SPAN_ID)
        return df


def make_tracing_backend(backend: str, endpoint: str | None = None) -> TracingBackend:
    """Factory for creating a TracingBackend from config values."""
    if backend == "phoenix":
        if not endpoint:
            raise ValueError("Phoenix backend requires traces_endpoint")
        return PhoenixBackend(endpoint)
    if backend == "mlflow":
        return MlflowBackend(endpoint)
    raise ValueError(f"Unknown traces_backend: {backend!r}. Supported: phoenix, mlflow")
