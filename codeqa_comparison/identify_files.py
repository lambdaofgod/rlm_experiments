"""Identify source files for each CodeQA context.

Commands:
    identify_repos      - find the GitHub repo for each context via code search
    clone_repos         - shallow-clone repos listed in repos.csv
    identify_files      - match context chunks to files in cloned repos
    assemble_contexts   - reassemble contexts with file boundary markers
"""

import csv
import logging
from pathlib import Path

import fire
from tqdm import tqdm

from clone import clone_repos
from common import load_codeqa, split_into_chunks
from file_match import read_repo_files, segments_from_chunks
from repo_search import RepoFinder, identify_repo


class Main:

    @classmethod
    def identify_repos(
        cls,
        output: str = "repos.csv",
        max_searches: int = 5,
        sleep: float = 6.0,
    ):
        """Identify the GitHub repo for each CodeQA context and write to CSV.

        Args:
            output: Path to output CSV file.
            max_searches: Max GitHub searches per context (only used for fallback).
            sleep: Seconds to sleep between API calls (rate limiting).
        """
        ds = load_codeqa()
        finder = RepoFinder()

        with open(output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["_id", "repo", "method"])

            for example in tqdm(ds, desc="Identifying repos"):
                ctx = example["context"]
                chunks = split_into_chunks(ctx)
                repo, method = identify_repo(
                    finder, ctx, chunks, max_searches=max_searches, sleep=sleep
                )
                writer.writerow([example["_id"], repo or "", method])
                f.flush()

                status = repo or "NOT FOUND"
                tqdm.write(f"  {example['_id']}: {status} ({method})")

    @staticmethod
    def clone_repos(
        repos_csv: str = "repos.csv",
        dest: str = "codeqa_repos",
    ):
        """Shallow-clone every unique repo from repos.csv into dest/."""
        clone_repos(repos_csv=repos_csv, dest=dest)

    @staticmethod
    def identify_files(
        repos_csv: str = "repos.csv",
        repos_dir: str = "codeqa_repos",
        output: str = "files.parquet",
        n_fragments: int = 5,
        fragment_size: int = 5,
    ):
        """Match context chunks to files in cloned repos.

        Args:
            repos_csv: CSV produced by identify_repos.
            repos_dir: Directory containing cloned repos.
            output: Path to output parquet file.
            n_fragments: Number of 5-line fragments to sample per chunk.
            fragment_size: Number of contiguous lines per fragment.
        """
        import pandas as pd

        repo_map = {}
        with open(repos_csv) as f:
            for row in csv.DictReader(f):
                repo = row["repo"].strip()
                if repo:
                    repo_map[row["_id"]] = repo

        ds = load_codeqa()
        repos_path = Path(repos_dir)
        file_cache: dict[str, dict[str, str]] = {}

        rows = []
        meta_rows = []
        for example in tqdm(ds, desc="Identifying files"):
            _id = example["_id"]
            repo = repo_map.get(_id)
            if not repo:
                meta_rows.append({"_id": _id, "repo": None, "n_segments": 0, "n_matched": 0, "ctx_lines": 0, "matched_lines": 0})
                continue

            repo_path = repos_path / repo
            if not repo_path.exists():
                meta_rows.append({"_id": _id, "repo": repo, "n_segments": 0, "n_matched": 0, "ctx_lines": 0, "matched_lines": 0})
                continue

            if repo not in file_cache:
                file_cache[repo] = read_repo_files(repo_path)
            repo_files = file_cache[repo]

            chunks = split_into_chunks(example["context"])
            segments = segments_from_chunks(chunks, repo_files, n_fragments, fragment_size)

            ctx_lines = example["context"].count("\n") + 1
            matched_lines = sum(s.text.count("\n") + 1 for s in segments if s.filename)
            n_matched = sum(1 for s in segments if s.filename)

            for idx, seg in enumerate(segments):
                rows.append({
                    "_id": _id,
                    "repo": repo,
                    "segment": idx,
                    "filename": seg.filename,
                    "match_method": seg.match_method,
                    "start_line": seg.start_line,
                    "end_line": seg.end_line,
                    "text": seg.text,
                })

            meta_rows.append({
                "_id": _id,
                "repo": repo,
                "n_segments": len(segments),
                "n_matched": n_matched,
                "ctx_lines": ctx_lines,
                "matched_lines": matched_lines,
            })

            tqdm.write(f"  {repo}: {n_matched}/{len(segments)} segments, {matched_lines}/{ctx_lines} lines ({matched_lines/ctx_lines:.0%})")

        df = pd.DataFrame(rows)
        df.to_parquet(output)

        meta_df = pd.DataFrame(meta_rows)
        meta_df.to_csv("file_matching_metadata.csv", index=False)

        n_examples = df["_id"].nunique()
        n_matched_examples = df.loc[df["filename"].notna(), "_id"].nunique()
        print(f"\nWrote {len(df)} segments ({n_examples} examples) to {output}")
        print(f"Wrote {len(meta_df)} rows to file_matching_metadata.csv")
        print(f"Examples with at least one file matched: {n_matched_examples}/{n_examples}")

    @staticmethod
    def assemble_contexts(
        segments_parquet: str = "files.parquet",
        output: str = "annotated_dataset.parquet",
        delimiter: str = "# --- {filename} ---",
        validate_strict: bool = False,
    ):
        """Reassemble contexts with file boundary markers.

        Args:
            segments_parquet: Path to parquet from identify_files.
            output: Path to output parquet file.
            delimiter: Format string with {filename} for file headers.
            validate_strict: If True, abort without writing when validation fails.
        """
        import pandas as pd

        from assemble import annotate_row, validate_annotated

        ds = load_codeqa()
        segments_df = pd.read_parquet(segments_parquet)
        original = {ex["_id"]: ex for ex in ds}

        rows = []
        errors = []
        grouped = segments_df.sort_values("segment").groupby("_id", sort=False)
        for _id, group in tqdm(grouped, desc="Assembling contexts"):
            result = annotate_row(group, delimiter)
            warning = validate_annotated(original[_id]["context"], result)
            if warning:
                tqdm.write(f"  WARNING {_id}: {warning}")
                errors.append(_id)

            row = dict(original[_id])
            row.update(result)
            rows.append(row)

        print(f"\n{len(rows)} examples, {len(errors)} validation errors")

        if errors and validate_strict:
            print("Strict validation failed. Output not written.")
            return

        out_df = pd.DataFrame(rows)
        out_df.to_parquet(output)
        print(f"Wrote {len(out_df)} annotated examples to {output}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    fire.Fire(Main())
