"""Clone repos listed in repos.csv."""

import csv
import subprocess
from pathlib import Path

from tqdm import tqdm


def clone_repos(
    repos_csv: str = "repos.csv",
    dest: str = "codeqa_repos",
):
    """Shallow-clone every unique repo from repos.csv into dest/.

    Repos are cloned into dest/owner/name. Already-cloned repos are skipped.

    Args:
        repos_csv: Path to the CSV produced by identify_repos.
        dest: Directory to clone into.
    """
    dest_dir = Path(dest)
    dest_dir.mkdir(parents=True, exist_ok=True)

    repos = set()
    with open(repos_csv) as f:
        for row in csv.DictReader(f):
            repo = row["repo"].strip()
            if repo:
                repos.add(repo)
    repos = sorted(repos)

    skipped = 0
    cloned = 0
    failed = 0

    pbar = tqdm(repos, desc="Cloning repos")
    for repo in pbar:
        pbar.set_postfix_str(repo)
        repo_dir = dest_dir / repo
        if repo_dir.exists():
            tqdm.write(f"  {repo}: already exists, skipping")
            skipped += 1
            continue

        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://github.com/{repo}.git"
        result = subprocess.run(
            ["git", "clone", "--depth", "1", url, str(repo_dir)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            tqdm.write(f"  FAILED {repo}: {result.stderr.strip()}")
            failed += 1
        else:
            cloned += 1

    print(f"\nDone: {cloned} cloned, {skipped} skipped, {failed} failed")
