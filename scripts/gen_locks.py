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
import configparser
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib

from pathlib import Path

from packaging.specifiers import InvalidSpecifier, SpecifierSet

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCKS_DIR = REPO_ROOT / "locks"
GECONPY_URL = "https://github.com/jessegrabowski/gEconpy.git"
GECONPY_CLONE = REPO_ROOT / ".cache" / "gEconpy"

# Python minor versions we build against, newest-first, each tagged with its release date. A
# candidate is only considered for a commit if it had been released by the commit's date -- otherwise
# conda-forge has no era-appropriate builds (e.g. a 2021 commit can't use py3.12, released 2023). The
# newest surviving candidate is tried first, falling back to older ones on a solve failure.
PYTHON_RELEASES = {
    "3.13": "2024-10-07",
    "3.12": "2023-10-02",
    "3.11": "2022-10-24",
    "3.10": "2021-10-04",
    "3.9": "2020-10-05",
}
PYTHON_CANDIDATES = list(PYTHON_RELEASES)

# Always present, regardless of era: build backends for both the setuptools/versioneer (1.x) and
# hatchling/hatch-vcs (2.x) worlds, the benchmark runner, and a couple of runtime deps that old
# commits relied on without declaring (joblib). The era-correct *runtime* deps are derived per
# commit from its own manifest by ``commit_deps``.
ALWAYS_DEPS = {"pip", "setuptools", "asv", "cloudpickle", "joblib"}

# Mid-2024 osx-arm64 openblas/libgfortran builds carry a duplicate-LC_RPATH bug that modern macOS
# dyld rejects. Float just the BLAS runtime to a current (fixed) build; it is not what is being
# benchmarked and does not perturb the date-locked numpy/scipy versions.
BLAS_FLOAT = {"libgfortran": "0d", "libgfortran5": "0d", "libopenblas": "0d"}

PLATFORMS = ["osx-arm64", "linux-64"]


def _norm_dep(spec):
    """Normalize a PEP 508 / requirement string to a conda-forge package name (or '')."""
    spec = spec.strip().strip("'\"")
    if not spec or spec.startswith("#"):
        return ""
    name = re.split(r"[<>=!~;\[ (]", spec)[0].strip()
    return name.lower().replace("_", "-")


def _dep_names(specs):
    return {name for name in map(_norm_dep, specs) if name}


def _show(commit, path):
    """Return ``path`` as of ``commit`` in the gEconpy clone, or "" if the commit has no such file."""
    try:
        return run(["git", "-C", str(GECONPY_CLONE), "show", f"{commit}:{path}"])
    except subprocess.CalledProcessError:
        return ""


def _pyproject(commit):
    try:
        return tomllib.loads(_show(commit, "pyproject.toml"))
    except tomllib.TOMLDecodeError:
        return {}


def commit_deps(commit):
    """Derive a commit's declared runtime deps as conda package names, across eras.

    2.x declares deps in ``pyproject.toml`` ``[project].dependencies``; 1.x uses ``setup.cfg``
    ``[options] install_requires`` (or, older still, an ``install_requires=[...]`` literal in
    ``setup.py``). The first source that yields anything wins.
    """
    deps = _dep_names(_pyproject(commit).get("project", {}).get("dependencies", []))
    if deps:
        return deps

    cfg = configparser.ConfigParser()
    try:
        cfg.read_string(_show(commit, "setup.cfg"))
        deps = _dep_names(cfg.get("options", "install_requires", fallback="").splitlines())
    except configparser.Error:
        pass
    if deps:
        return deps

    match = re.search(r"install_requires\s*=\s*\[([^\]]*)\]", _show(commit, "setup.py"), re.S)
    if match is None:
        return set()
    return _dep_names(re.findall(r"['\"]([^'\"]+)['\"]", match.group(1)))


def build_deps(commit):
    """Build-backend packages from the commit's ``[build-system].requires``; fall back to setuptools.

    1.x commits (setup.py/cfg) have no ``[build-system]``, so plain setuptools handles the build; 2.x
    lists hatchling/hatch-vcs (and some versions versioneer) here. Reading it per commit avoids
    injecting newer backends into old eras where they had no conda-forge builds yet.
    """
    requires = _pyproject(commit).get("build-system", {}).get("requires", [])
    return _dep_names(requires) or {"setuptools"}


def full_deps(commit):
    """Era-correct runtime deps, the commit's build backend, and the always-present bench deps."""
    return commit_deps(commit) | build_deps(commit) | ALWAYS_DEPS


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kw).stdout


def ensure_clone():
    if not GECONPY_CLONE.exists():
        GECONPY_CLONE.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", GECONPY_URL, str(GECONPY_CLONE)])
    run(["git", "-C", str(GECONPY_CLONE), "fetch", "--all", "--tags", "--prune"])


def full_hash(commit):
    """Resolve a commit name to its full hash in the gEconpy clone, raising ValueError if git can't."""
    command = ["git", "-C", str(GECONPY_CLONE), "rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}"]
    try:
        return run(command).strip()
    except subprocess.CalledProcessError:
        raise ValueError(f"{commit!r} is not a commit in {GECONPY_CLONE}") from None


def resolve_commits(args):
    if args.commits:
        return [full_hash(commit.strip()) for commit in args.commits]
    commits = run(["git", "-C", str(GECONPY_CLONE), "rev-list", "--reverse", args.range or "main"]).split()
    if args.steps and args.steps < len(commits):
        step = len(commits) / args.steps
        commits = [commits[int(i * step)] for i in range(args.steps)]
    return commits


def commit_date(commit):
    return run(["git", "-C", str(GECONPY_CLONE), "show", "-s", "--format=%cs", commit]).strip()


def candidate_pythons(commit):
    """Newest-first Pythons that satisfy the commit's requires-python AND existed by its date.

    The newest surviving candidate is tried first; the solve is retried with older ones on failure.
    Date-capping matters for old commits: a 2021 commit can only use a Python that had conda-forge
    builds in 2021, regardless of what the (often absent) requires-python says.
    """
    spec = _pyproject(commit).get("project", {}).get("requires-python", "")
    try:
        allowed = SpecifierSet(spec)
    except InvalidSpecifier:
        allowed = SpecifierSet()
    date = commit_date(commit)
    candidates = [c for c in PYTHON_CANDIDATES if c in allowed and PYTHON_RELEASES[c] <= date]
    return candidates or ["3.9"]


def manifest(date, python, deps):
    dep_lines = "\n".join(f'{d} = "*"' for d in sorted(deps))
    platforms = ", ".join(f'"{p}"' for p in PLATFORMS)
    blas = "\n".join(f'{pkg} = "{when}"' for pkg, when in BLAS_FLOAT.items())
    return (
        "[workspace]\n"
        'name = "gEconpy-bench-locked"\n'
        'channels = ["conda-forge"]\n'
        f"platforms = [{platforms}]\n"
        f'exclude-newer = "{date}"\n\n'
        "[dependencies]\n"
        f'python = "{python}.*"\n'
        f"{dep_lines}\n\n"
        "[exclude-newer]\n"
        f"{blas}\n"
    )


def existing_lock(commit):
    """Return the lock path for a commit if one was already generated (any python), else None."""
    matches = list(LOCKS_DIR.glob(f"*_{commit[:10]}_py*.lock"))
    return matches[0] if matches else None


def generate(commit):
    date = commit_date(commit)
    if existing_lock(commit):
        print(f"  skip   {date} {commit[:10]} (lock exists)")
        return True
    pythons = candidate_pythons(commit)
    deps = full_deps(commit)
    last_err = ""
    for python in pythons:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            (tmp / "pixi.toml").write_text(manifest(date, python, deps))
            try:
                run(["pixi", "lock", "--manifest-path", str(tmp / "pixi.toml")])
            except subprocess.CalledProcessError as exc:
                last_err = (exc.stderr.strip().splitlines() or [""])[-1].strip()
                continue
            out = LOCKS_DIR / f"{date}_{commit[:10]}_py{python}.lock"
            shutil.copy(tmp / "pixi.lock", out)
            print(f"  locked {date} {commit[:10]} py{python}")
            return True
    print(f"  FAIL   {date} {commit[:10]} (tried py{', py'.join(pythons)}): {last_err}")
    return False


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
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
