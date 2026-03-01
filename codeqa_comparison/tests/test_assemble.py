"""Tests for assemble.py -- context reconstruction with file headers."""

import pandas as pd

from assemble import annotate_context, annotate_row


def test_annotate_context_basic():
    """Segments with filenames get headers, None segments get none."""
    segments = [
        {"filename": "src/foo.py", "text": "import os\n\ndef foo():\n    pass"},
        {"filename": None, "text": "# some unknown chunk"},
        {"filename": "src/bar.py", "text": "import sys\n\ndef bar():\n    pass"},
    ]
    result = annotate_context(segments, delimiter="# --- {filename} ---")
    lines = result.split("\n")
    assert lines[0] == "# --- src/foo.py ---"
    assert "def foo():" in result
    assert "# --- src/bar.py ---" in result
    # None segment should have no header
    assert "# --- None ---" not in result
    assert "# some unknown chunk" in result


def test_annotate_context_mpmath():
    """Integration test: annotate_context against real mpmath data from files.parquet."""
    df = pd.read_parquet("files.parquet")
    mpmath = df[df["_id"] == "66fa208bbb02136c067c5fc1"].sort_values("segment")
    segments = [
        {"filename": row["filename"], "text": row["text"]}
        for _, row in mpmath.iterrows()
    ]

    delimiter = "# --- {filename} ---"
    result = annotate_context(segments, delimiter=delimiter)
    lines = result.split("\n")

    # 52 matched segments produce 52 header lines
    header_lines = [l for l in lines if l.startswith("# --- ") and l.endswith(" ---")]
    assert len(header_lines) == 52

    # First line is the header for the first segment
    assert lines[0] == "# --- mpmath/identification.py ---"

    # Last line is from the last segment's text, not a header
    assert lines[-1] == "        int_types = (int, MPZ_TYPE)"

    # No headers for unmatched (None filename) segments
    assert "# --- None ---" not in result

    # All 51 distinct filenames appear in headers
    expected_filenames = set(mpmath["filename"].dropna())
    assert len(expected_filenames) == 51
    for fn in expected_filenames:
        assert f"# --- {fn} ---" in result


def test_annotate_row_basic():
    """annotate_row returns annotated context and metadata for one example."""
    segments = pd.DataFrame(
        {
            "filename": ["src/foo.py", None, "src/bar.py"],
            "text": ["def foo(): pass", "# orphan", "def bar(): pass"],
            "repo": ["owner/repo"] * 3,
        }
    )
    delimiter = "# --- {filename} ---"
    row = annotate_row(segments, delimiter=delimiter)

    assert "# --- src/foo.py ---" in row["context"]
    assert "# --- None ---" not in row["context"]
    assert row["n_segments"] == 3
    assert row["n_matched"] == 2
    assert row["repo"] == "owner/repo"
    assert isinstance(row["instruction_suffix"], str)
    assert len(row["instruction_suffix"]) > 0
