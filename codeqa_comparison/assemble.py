"""Reassemble CodeQA contexts with explicit file boundary markers."""

import pandas as pd


def annotate_context(segments: list[dict], delimiter: str) -> str:
    """Reconstruct context with file headers before each matched segment.

    For each segment with a non-None filename, prepend a header line using
    *delimiter* (which must contain ``{filename}``). Unmatched segments
    (filename is None/NaN) are included without a header.
    """
    parts: list[str] = []
    for seg in segments:
        fn = seg["filename"]
        if fn is not None and (not isinstance(fn, float) or fn == fn):
            parts.append(delimiter.format(filename=fn))
        parts.append(seg["text"])
    return "\n".join(parts)


def make_instruction_suffix(delimiter: str) -> str:
    """Return a short string explaining the delimiter convention."""
    example = delimiter.format(filename="path/to/file.py")
    return f"File boundaries in the context are marked with lines like: {example}"


def annotate_row(segments_df: pd.DataFrame, delimiter: str) -> dict:
    """Annotate a single example's segments into a context with metadata.

    Parameters
    ----------
    segments_df : pd.DataFrame
        DataFrame of segments for one example, with columns
        ``filename``, ``text``, and ``repo``.
    delimiter : str
        Format string containing ``{filename}`` used for file headers.

    Returns
    -------
    dict with keys: context, instruction_suffix, n_segments, n_matched, repo.
    """
    segments = segments_df.to_dict(orient="records")
    context = annotate_context(segments, delimiter)
    instruction_suffix = make_instruction_suffix(delimiter)
    n_segments = len(segments_df)
    n_matched = int(segments_df["filename"].notna().sum())
    repo = segments_df["repo"].iloc[0]
    return {
        "context": context,
        "instruction_suffix": instruction_suffix,
        "n_segments": n_segments,
        "n_matched": n_matched,
        "repo": repo,
    }


def validate_annotated(original_context: str, annotated: dict) -> str | None:
    """Check the annotated output for a single example.

    Verifies that the annotated context has exactly as many extra lines
    as headers were inserted (n_matched), meaning no text was lost or
    duplicated. Returns a warning string on failure, None on success.
    """
    orig_lines = original_context.count("\n") + 1
    new_lines = annotated["context"].count("\n") + 1
    expected = orig_lines + annotated["n_matched"]
    if new_lines != expected:
        return f"line count {new_lines} != expected {expected} (original {orig_lines} + {annotated['n_matched']} headers)"
    return None
