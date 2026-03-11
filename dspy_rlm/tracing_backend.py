"""Abstract tracing backend and Phoenix implementation.

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


def make_tracing_backend(backend: str, endpoint: str) -> TracingBackend:
    """Factory for creating a TracingBackend from config values."""
    if backend == "phoenix":
        return PhoenixBackend(endpoint)
    raise ValueError(f"Unknown traces_backend: {backend!r}. Supported: phoenix")
