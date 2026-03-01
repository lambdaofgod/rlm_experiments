"""Match context chunks to files in cloned repos."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from common import FileChunk


@dataclass
class Segment:
    filename: str | None
    match_method: str | None
    text: str
    start_line: int
    end_line: int


def read_repo_files(repo_path: Path) -> dict[str, str]:
    """Read all text files in a repo into {relative_path: content}."""
    result = {}
    for p in repo_path.rglob("*"):
        if not p.is_file():
            continue
        if ".git" in p.parts:
            continue
        try:
            content = p.read_text(errors="ignore")
            result[str(p.relative_to(repo_path))] = content
        except Exception:
            continue
    return result


def sample_fragments(
    chunk: FileChunk, n_fragments: int, fragment_size: int = 5
) -> list[str]:
    """Sample contiguous fragments of fragment_size lines from a chunk.

    Fragments are evenly spaced across the chunk.
    """
    lines = chunk.lines
    if len(lines) < fragment_size:
        return ["\n".join(lines)] if lines else []

    max_start = len(lines) - fragment_size
    if max_start == 0:
        return ["\n".join(lines[:fragment_size])]

    step = max(1, max_start // n_fragments)
    fragments = []
    for i in range(0, max_start + 1, step):
        fragments.append("\n".join(lines[i : i + fragment_size]))
        if len(fragments) >= n_fragments:
            break
    return fragments


def _match_by_hint(hint: str, repo_files: dict[str, str]) -> str | None:
    """Find a repo file whose path ends with the given hint basename."""
    matches = [p for p in repo_files if p.endswith("/" + hint) or p == hint]
    if len(matches) == 1:
        return matches[0]
    return None


def match_chunk(
    chunk: FileChunk,
    repo_files: dict[str, str],
    n_fragments: int,
    fragment_size: int = 5,
) -> tuple[str | None, str | None]:
    """Match a chunk to a file by grepping fragments and filename hint.

    Tries both fragment-based matching (from repo) and hint-based matching
    (from #include). Fragment match has precedence; if both agree the method
    is "fragment+hint".

    Returns (file_path, match_method) where match_method is one of:
    "fragment", "hint", "fragment+hint", or None.
    """
    fragment_match = None
    fragments = sample_fragments(chunk, n_fragments, fragment_size)
    if fragments:
        hits: Counter[str] = Counter()
        for frag in fragments:
            for path, content in repo_files.items():
                if frag in content:
                    hits[path] += 1
        if hits:
            fragment_match = hits.most_common(1)[0][0]

    hint_match = None
    if chunk.filename_hint:
        hint_match = _match_by_hint(chunk.filename_hint, repo_files)

    if fragment_match and hint_match:
        if fragment_match == hint_match:
            return fragment_match, "fragment+hint"
        return fragment_match, "fragment"
    if fragment_match:
        return fragment_match, "fragment"
    if hint_match:
        return hint_match, "hint"
    return None, None


def segments_from_chunks(
    chunks: list[FileChunk],
    repo_files: dict[str, str],
    n_fragments: int,
    fragment_size: int = 5,
) -> list[Segment]:
    """Match each chunk to a file, then merge consecutive same-file chunks."""
    labeled: list[tuple[str | None, str | None, FileChunk]] = []
    for chunk in chunks:
        filename, method = match_chunk(chunk, repo_files, n_fragments, fragment_size)
        labeled.append((filename, method, chunk))

    segments: list[Segment] = []
    for filename, method, chunk in labeled:
        if segments and segments[-1].filename == filename:
            segments[-1].text += "\n" + "\n".join(chunk.lines)
            segments[-1].end_line = chunk.end_line
        else:
            segments.append(Segment(
                filename=filename,
                match_method=method,
                text="\n".join(chunk.lines),
                start_line=chunk.start_line,
                end_line=chunk.end_line,
            ))
    return segments
