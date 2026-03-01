"""Shared types and utilities for CodeQA processing."""

import re
from dataclasses import dataclass, field

from datasets import load_dataset


@dataclass
class FileChunk:
    start_line: int
    end_line: int
    lines: list[str]
    definitions: list[str] = field(default_factory=list)
    filename_hint: str | None = None


def detect_language(ctx: str) -> str:
    """Detect the primary programming language of a context.

    Looks at class/struct/interface definitions and import patterns to
    distinguish languages. Returns "py", "cpp", "ts", or "unknown".
    """
    lines = ctx.split("\n")
    py = 0
    cpp = 0
    ts = 0

    for line in lines:
        # Class definitions
        # Python: class Foo:  or  class Foo(Bar):
        if re.match(r"^class\s+\w+.*:\s*$", line):
            py += 1
        # C/C++: class Foo {  or  struct Foo {  or  class Foo : public Bar
        elif re.match(r"^\s*(?:class|struct|enum)\s+\w+.*[{]", line):
            cpp += 1
        elif re.match(r"^\s*(?:class|struct)\s+\w+\s*:\s*public\b", line):
            cpp += 1
        # TS/JS: interface Foo {  or  export class Foo {
        elif re.match(r"^\s*(?:export\s+)?(?:interface|class)\s+\w+.*[{]", line):
            ts += 1

        # Import patterns
        # C/C++: #include
        if re.match(r"^\s*#include\s+[<\"]", line):
            cpp += 1
        # Python: import foo / from foo import bar
        elif re.match(r"^(?:import|from)\s+\w+", line):
            py += 1
        # TS/JS: import ... from "..."  or  import "..."
        elif re.match(r"^\s*import\s+.*\sfrom\s+['\"]", line):
            ts += 1
        elif re.match(r"^\s*import\s+['\"]", line):
            ts += 1

    scores = {"py": py, "cpp": cpp, "ts": ts}
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "unknown"
    return best


def _is_cpp_comment_line(line: str) -> bool:
    """Check if a line looks like part of a C/C++ comment block."""
    s = line.strip()
    return (
        s.startswith("//")
        or s.startswith("/*")
        or s.startswith("*")
        or s.endswith("*/")
    )


def _find_cpp_boundaries(lines: list[str]) -> list[int]:
    """Find C++ file boundaries: #include after a comment block + blank line.

    Returns the index of the first line of the comment block for each boundary.
    """
    boundaries = []
    prev_was_include = False
    for i, line in enumerate(lines):
        is_include = bool(re.match(r"^\s*#include\s", line))
        if is_include and not prev_was_include:
            # Walk back over blank lines
            j = i - 1
            while j >= 0 and not lines[j].strip():
                j -= 1
            if j < 0:
                prev_was_include = is_include
                continue
            if not _is_cpp_comment_line(lines[j]):
                prev_was_include = is_include
                continue
            # Walk back over the comment block
            while j >= 0 and _is_cpp_comment_line(lines[j]):
                j -= 1
            # j+1 is the first line of the comment block
            comment_start = j + 1
            # Only treat as boundary if comment block is preceded by a blank
            # line (or is at the very start)
            if comment_start == 0 or not lines[comment_start - 1].strip():
                boundaries.append(comment_start)
        prev_was_include = is_include
    return boundaries


def split_into_chunks(ctx: str, min_chunk_lines: int = 20) -> list[FileChunk]:
    """Split a concatenated context into chunks at likely file boundaries."""
    lines = ctx.split("\n")
    lang = detect_language(ctx)
    boundary_indices = [0]

    # C++ specific: comment block + #include section
    if lang == "cpp":
        for idx in _find_cpp_boundaries(lines):
            if idx > 0 and idx - boundary_indices[-1] >= min_chunk_lines:
                boundary_indices.append(idx)

    for i in range(1, len(lines)):
        line = lines[i]

        if i > 0 and lines[i - 1].strip():
            continue

        # Markdown file-path headers (## [path]) are always boundaries
        is_file_header = bool(re.match(r"^##\s+\[.+\]\(", line))

        is_preamble = (
            line.startswith('"""')
            or line.startswith("'''")
            or line.startswith("from __future__")
            or line.startswith("#!")
            or line.startswith("# -*-")
            or re.match(r"^cmake_minimum_required", line)
            or re.match(r"^#\s+\w", line)
        )
        if not is_preamble and not is_file_header:
            continue

        if not is_file_header:
            j = i - 1
            while j >= 0 and not lines[j].strip():
                j -= 1
            if j >= 0:
                prev = lines[j].strip()
                if (
                    prev.endswith(":")
                    or prev.startswith("def ")
                    or prev.startswith("class ")
                ):
                    continue
                if prev.endswith('"""') or prev.endswith("'''"):
                    continue

        if is_file_header or i - boundary_indices[-1] >= min_chunk_lines:
            boundary_indices.append(i)

    # Deduplicate and sort
    boundary_indices = sorted(set(boundary_indices))

    chunks = []
    for k in range(len(boundary_indices)):
        start = boundary_indices[k]
        end = boundary_indices[k + 1] if k + 1 < len(boundary_indices) else len(lines)
        chunk_lines = lines[start:end]

        defs = []
        filename_hint = None
        for cl in chunk_lines:
            # Python: class Foo, def bar
            m = re.match(r"^(class|def)\s+(\w+)", cl)
            if m:
                defs.append(m.group(2))
                continue
            # C/C++: struct Foo, class Foo, enum Foo (ignore standard-lib types)
            m = re.match(r"^\s*(?:struct|class|enum)\s+(\w+)", cl)
            if m and len(m.group(1)) > 3:
                defs.append(m.group(1))
            # C/C++: extract filename hint from first #include
            if filename_hint is None and lang == "cpp":
                m = re.match(r'^\s*#include\s+["<](.+?)[">]', cl)
                if m:
                    filename_hint = m.group(1)

        chunks.append(
            FileChunk(
                start_line=start,
                end_line=end,
                lines=chunk_lines,
                definitions=defs,
                filename_hint=filename_hint,
            )
        )

    return chunks


def load_codeqa():
    ds = load_dataset("THUDM/LongBench-v2", split="train")
    return ds.filter(lambda ex: ex["domain"] == "Code Repository Understanding")
