"""Identify the GitHub repo for a CodeQA context."""

import json
import logging
import re
import subprocess
import time
from collections import Counter

from github import Github, GithubException
from pydantic import BaseModel, ConfigDict, Field
from retry import retry

from common import FileChunk

log = logging.getLogger(__name__)

_GITHUB_URL_RE = re.compile(r"github\.com/([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)")


class RepoFinder(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    gh: Github = Field(default_factory=Github)
    cache: dict[str, bool] = Field(default_factory=dict)

    def repo_exists(self, full_name: str) -> bool:
        """Check whether a GitHub repo exists."""
        if full_name not in self.cache:
            try:
                self.gh.get_repo(full_name)
                self.cache[full_name] = True
            except GithubException:
                self.cache[full_name] = False
        return self.cache[full_name]

    def repo_from_url(self, ctx: str) -> str | None:
        """Extract owner/repo from a GitHub URL found anywhere in the context."""
        matches = _GITHUB_URL_RE.findall(ctx)
        if not matches:
            return None
        cleaned = [m.rstrip(".") for m in matches]
        for repo, _count in Counter(cleaned).most_common():
            if self.repo_exists(repo):
                return repo
        return None


class GitHubRateLimitError(Exception):
    pass


def repo_from_readme(chunks: list[FileChunk]) -> str | None:
    """Try to find a repo name from a README-like chunk header.

    Looks for chunks that start with a markdown heading (# ProjectName)
    and checks if the name looks like a plausible package/repo name.
    """
    for chunk in chunks:
        first = chunk.lines[0].strip() if chunk.lines else ""
        m = re.match(r"^#\s+(\S+)", first)
        if not m:
            continue
        name = m.group(1).lower()
        rest = "\n".join(chunk.lines[1:50])
        if (
            f"import {name}" in rest
            or f"from {name}" in rest
            or f"pip install {name}" in rest
        ):
            return name
    return None


def _sanitize_query(s: str) -> str:
    """Keep only characters safe for GitHub code search exact-match queries."""
    return re.sub(r"[^a-zA-Z0-9 _.,:;=+\-/#@()<>]", "", s).strip()


def pick_query_line(chunk_lines: list[str], n: int = 20) -> str:
    """Pick the longest non-blank line from the first n lines of a chunk.

    Sanitizes to characters safe for GitHub's search query parser.
    """
    candidates = []
    for l in chunk_lines[:n]:
        stripped = l.strip()
        if stripped.startswith(("//", "#", "/*", "*", "```")):
            continue
        cleaned = _sanitize_query(l)
        if len(cleaned) >= 15:
            candidates.append(cleaned)
    if not candidates:
        return ""
    return max(candidates, key=len)


_BLACKLISTED_REPOS = {
    "AmanPriyanshu/long-context-understanding-benchmark-raw-files-only",
}


@retry(GitHubRateLimitError, tries=5, delay=60, backoff=2)
def _gh_code_search(query: str, sleep: float = 6.0) -> str | None:
    """Run a GitHub code search query and return the top repo full_name."""
    result = subprocess.run(
        [
            "gh",
            "api",
            "search/code",
            "-X",
            "GET",
            "-f",
            f'q="{query}"',
            "-f",
            "per_page=5",
        ],
        capture_output=True,
        text=True,
    )
    time.sleep(sleep)
    stderr = result.stderr.strip()
    if "rate limit" in stderr.lower() or "HTTP 403" in stderr:
        log.warning("rate limited, will retry...")
        raise GitHubRateLimitError(stderr)
    if result.returncode != 0:
        log.warning("gh search failed (query=%r): %s", query, stderr)
        return None
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        log.warning("bad JSON from gh")
        return None
    for item in data.get("items", []):
        repo = item["repository"]["full_name"]
        if repo not in _BLACKLISTED_REPOS:
            return repo
    return None


def search_chunk(chunk: FileChunk, sleep: float = 6.0) -> str | None:
    """Search GitHub code for a chunk and return the repo full_name, or None."""
    query = pick_query_line(chunk.lines)
    if not query:
        return None
    return _gh_code_search(query, sleep=sleep)


def search_repo(
    chunks: list[FileChunk], max_searches: int = 5, sleep: float = 6.0
) -> str | None:
    """Search a few chunks on GitHub and return the most common repo name.

    Tries line-based search across sampled chunks, then definition-pair
    searches as a fallback. Uses majority voting across all results.
    """
    step = max(1, len(chunks) // max_searches)
    sampled = chunks[::step][:max_searches]

    repos = []
    for chunk in sampled:
        repo = search_chunk(chunk, sleep=sleep)
        if repo:
            repos.append(repo)

    budget = max_searches - len(repos)
    for chunk in sampled[:budget]:
        long_defs = [d for d in chunk.definitions if len(d) >= 8]
        if len(long_defs) >= 2:
            query = " ".join(long_defs[:3])
            repo = _gh_code_search(query, sleep=sleep)
            if repo:
                repos.append(repo)

    if not repos:
        return None
    return Counter(repos).most_common(1)[0][0]


def identify_repo(
    finder: RepoFinder,
    ctx: str,
    chunks: list[FileChunk],
    max_searches: int = 10,
    sleep: float = 6.0,
) -> tuple[str | None, str]:
    """Identify the repo for a context. Returns (repo, method).

    Tries in order:
    1. GitHub URL in the context text
    2. README heading that matches an import
    3. GitHub code search (fallback)
    """
    repo = finder.repo_from_url(ctx)
    if repo:
        return repo, "url"

    repo = repo_from_readme(chunks)
    if repo:
        return repo, "readme"

    repo = search_repo(chunks, max_searches=max_searches, sleep=sleep)
    if repo:
        return repo, "search"

    return None, "not_found"
