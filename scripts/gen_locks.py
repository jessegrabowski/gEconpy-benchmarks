#!/usr/bin/env python
"""Generate one date-locked pixi lockfile per gEconpy commit.

For each commit, the conda-forge dependency tree is resolved as it existed on the commit's date
(pixi's ``exclude-newer``), and an era-appropriate Python is chosen from the commit's own
``requires-python``. The resulting lock is written to ``locks/<date>_<shorthash>.lock``. Existing
locks are skipped, so the job is resumable. gEconpy itself is never locked here — it is installed
from a checkout of the commit at run time (see ``run_history.py``).

Usage::

    python scripts/gen_locks.py                       # every commit on main
    python scripts/gen_locks.py --range v0.1..main    # a git range
    python scripts/gen_locks.py --steps 40            # evenly spaced subset
    python scripts/gen_locks.py --commits abc123 def456
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCKS_DIR = REPO_ROOT / "locks"
GECONPY_URL = "https://github.com/jessegrabowski/gEconpy.git"
GECONPY_CLONE = REPO_ROOT / ".cache" / "gEconpy"

# Python minor versions we are willing to build against, newest first. The newest that satisfies a
# commit's requires-python is chosen; pixi's exclude-newer keeps the actual build era-appropriate.
PYTHON_CANDIDATES = ["3.13", "3.12", "3.11", "3.10"]

# Dependency set mirrored from pixi.toml. Versions are intentionally unconstrained here — the date
# lock, not a floor, determines what gets installed.
RUNTIME_DEPS = [
    "pip", "hatchling", "hatch-vcs", "setuptools",
    "numpy", "scipy", "numba", "pandas", "matplotlib", "sympy",
    "pytensor", "pymc", "sympytensor", "preliz", "pymc-extras", "better-optimize",
    "pyparsing", "xarray", "networkx", "ipython",
    "asv",
]
PLATFORMS = ["osx-arm64", "linux-64"]


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kw).stdout


def ensure_clone():
    if not GECONPY_CLONE.exists():
        GECONPY_CLONE.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", GECONPY_URL, str(GECONPY_CLONE)])
    run(["git", "-C", str(GECONPY_CLONE), "fetch", "--all", "--tags", "--prune"])


def resolve_commits(args):
    base = ["git", "-C", str(GECONPY_CLONE), "rev-list"]
    if args.commits:
        return [c.strip() for c in args.commits]
    rng = args.range or "main"
    out = run([*base, "--reverse", rng])
    commits = out.split()
    if args.steps and args.steps < len(commits):
        step = len(commits) / args.steps
        commits = [commits[int(i * step)] for i in range(args.steps)]
    return commits


def commit_date(commit):
    return run(["git", "-C", str(GECONPY_CLONE), "show", "-s", "--format=%cs", commit]).strip()


def pick_python(commit):
    """Newest candidate Python satisfying the commit's requires-python (defaults to 3.12)."""
    try:
        text = run(["git", "-C", str(GECONPY_CLONE), "show", f"{commit}:pyproject.toml"])
        spec = tomllib.loads(text).get("project", {}).get("requires-python", "")
    except (subprocess.CalledProcessError, tomllib.TOMLDecodeError, KeyError):
        spec = ""
    floor = re.search(r">=?\s*3\.(\d+)", spec)
    min_minor = int(floor.group(1)) if floor else 12
    # PYTHON_CANDIDATES is newest-first; return the newest that clears the floor.
    for cand in PYTHON_CANDIDATES:
        if int(cand.split(".")[1]) >= min_minor:
            return cand
    return "3.12"


def manifest(date, python):
    deps = "\n".join(f'{d} = "*"' for d in RUNTIME_DEPS)
    platforms = ", ".join(f'"{p}"' for p in PLATFORMS)
    return (
        "[workspace]\n"
        'name = "gEconpy-bench-locked"\n'
        'channels = ["conda-forge"]\n'
        f"platforms = [{platforms}]\n"
        f'exclude-newer = "{date}"\n\n'
        "[dependencies]\n"
        f'python = "{python}.*"\n'
        f"{deps}\n"
    )


def lock_path(date, commit):
    return LOCKS_DIR / f"{date}_{commit[:10]}.lock"


def generate(commit):
    date = commit_date(commit)
    out = lock_path(date, commit)
    if out.exists():
        print(f"  skip   {date} {commit[:10]} (lock exists)")
        return True
    python = pick_python(commit)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "pixi.toml").write_text(manifest(date, python))
        try:
            subprocess.run(
                ["pixi", "lock", "--manifest-path", str(tmp / "pixi.toml")],
                check=True, text=True, capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            print(f"  FAIL   {date} {commit[:10]} py{python}: {exc.stderr.strip().splitlines()[-1:]}")
            return False
        shutil.copy(tmp / "pixi.lock", out)
    print(f"  locked {date} {commit[:10]} py{python}")
    return True


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--range", help="git rev-list range (default: main)")
    p.add_argument("--steps", type=int, help="evenly-spaced subset of the range")
    p.add_argument("--commits", nargs="+", help="explicit commit hashes")
    args = p.parse_args(argv)

    LOCKS_DIR.mkdir(exist_ok=True)
    ensure_clone()
    commits = resolve_commits(args)
    print(f"Generating locks for {len(commits)} commit(s)")
    failures = sum(not generate(c) for c in commits)
    if failures:
        print(f"\n{failures} commit(s) could not be locked (logged above).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
