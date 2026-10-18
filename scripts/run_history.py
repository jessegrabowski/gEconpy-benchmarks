#!/usr/bin/env python
"""Benchmark gEconpy across history, one date-locked environment per commit.

For each commit with a lock in ``locks/`` (produced by ``gen_locks.py``), this reconstructs the
date-locked pixi environment, installs that commit's gEconpy into it, and runs the suite via
``asv run --python=<locked-python> --set-commit-hash <commit>`` — so the result is attributed to the
commit while running against era-appropriate dependencies. Commits already present in
``.asv/results`` are skipped, so the sweep is resumable.

Usage::

    python scripts/run_history.py                    # every commit that has a lock
    python scripts/run_history.py --steps 40
    python scripts/run_history.py --commits abc123 def456
"""

import argparse
import json
import platform
import shutil
import subprocess
import sys

from pathlib import Path

import gen_locks  # sibling module; provides clone + manifest helpers

REPO_ROOT = gen_locks.REPO_ROOT
ENVS_CACHE = REPO_ROOT / ".cache" / "envs"
RESULTS_DIR = REPO_ROOT / ".asv" / "results"


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, text=True, **kw)


def machine_name():
    cfg = Path.home() / ".asv-machine.json"
    if cfg.exists():
        data = json.loads(cfg.read_text())
        names = [k for k in data if k != "version"]
        if names:
            return names[0]
    return platform.node()


def already_done(commit, machine):
    results = RESULTS_DIR / machine
    if not results.exists():
        return False
    prefix = commit[:8]
    return any(f.name.startswith(prefix) for f in results.glob("*.json"))


def locked_commits():
    """Map commit-prefix -> (lock path, python), from ``<date>_<shorthash>_py<ver>.lock``."""
    out = {}
    for lock in gen_locks.LOCKS_DIR.glob("*_py*.lock"):
        _date, short, py = lock.stem.split("_")
        out[short] = (lock, py.removeprefix("py"))
    return out


def env_dir(commit):
    return ENVS_CACHE / commit[:10]


def locked_manifest(commit, lock, python):
    """Return the manifest ``lock`` was solved from, rebuilding it for locks that predate stored manifests."""
    stored = gen_locks.manifest_path(lock)
    if stored.exists():
        return stored.read_text()
    return gen_locks.manifest(gen_locks.commit_date(commit), python, gen_locks.full_deps(commit))


def build_env(commit, lock, python):
    """Reconstruct the date-locked env from its lock and return the env's python path."""
    work = env_dir(commit)
    work.mkdir(parents=True, exist_ok=True)
    (work / "pixi.toml").write_text(locked_manifest(commit, lock, python))
    (work / "pixi.lock").write_bytes(lock.read_bytes())
    run(["pixi", "install", "--locked", "--manifest-path", str(work / "pixi.toml")])
    py = work / ".pixi" / "envs" / "default" / "bin" / "python"
    if not py.exists():
        raise FileNotFoundError(f"locked env python not found at {py}")
    return py


def install_geconpy(commit, python):
    run(["git", "-C", str(gen_locks.GECONPY_CLONE), "checkout", "--quiet", "--force", commit])
    # Force a current asv into the target env: it's the benchmark harness (not the thing measured),
    # and the date-locked asv in pre-2024 envs predates the asv/asv_runner split, which the current
    # orchestrator requires. Installed on top of the locked env so no re-lock is needed.
    run([str(python), "-m", "pip", "install", "-U", "asv>=0.6.4", "--quiet"])
    run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            str(gen_locks.GECONPY_CLONE),
            "--no-build-isolation",
            "--quiet",
            "--no-deps",
        ]
    )


def benchmark(commit, python, machine, asv_args):
    run(
        [
            "asv",
            "run",
            f"--python={python}",
            "--set-commit-hash",
            commit,
            "--machine",
            machine,
            "--show-stderr",
            *asv_args,
        ],
        cwd=str(REPO_ROOT),
    )


def installable(commit):
    """Return True if the commit has a pip-installable packaging file (setup.py / pyproject) at its root.

    Pre-packaging commits (early-2021..~2022 prototype era) ran from source and can't be installed;
    checking the git tree first skips them instantly instead of building an env that then fails.
    """
    try:
        tree = gen_locks.run(["git", "-C", str(gen_locks.GECONPY_CLONE), "ls-tree", "--name-only", commit])
    except subprocess.CalledProcessError:
        return False
    return bool(set(tree.split()) & {"setup.py", "pyproject.toml"})


def process(commit, lock, python, machine, asv_args, rerun=False):
    if not rerun and already_done(commit, machine):
        print(f"  skip   {commit[:10]} (results exist)")
        return True
    if not installable(commit):
        print(f"  skip   {commit[:10]} (pre-packaging: no setup.py/pyproject)")
        return True
    try:
        env_python = build_env(commit, lock, python)
        install_geconpy(commit, env_python)
        benchmark(commit, env_python, machine, asv_args)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"  FAIL   {commit[:10]}: {exc}")
        return False
    finally:
        # Reclaim the per-commit env; it's reconstructable from the lock. Keeps a 200+ commit sweep
        # from accumulating one full conda env per commit on disk.
        shutil.rmtree(env_dir(commit), ignore_errors=True)
    print(f"  done   {commit[:10]}")
    return True


def resolve(args, locks):
    """Return a list of (commit, lock_path, python) to benchmark, in history order."""
    gen_locks.ensure_clone()
    if args.commits:
        requested = [gen_locks.full_hash(commit) for commit in args.commits]
        wanted = [(full, *locks[full[:10]]) for full in requested if full[:10] in locks]
    else:
        order = gen_locks.run(
            ["git", "-C", str(gen_locks.GECONPY_CLONE), "rev-list", "--reverse", args.range or "main"]
        ).split()
        wanted = [(full, *locks[full[:10]]) for full in order if full[:10] in locks]
    if args.steps and args.steps < len(wanted):
        step = len(wanted) / args.steps
        wanted = [wanted[int(i * step)] for i in range(args.steps)]
    return wanted


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--range", help="git rev-list range (default: main)")
    p.add_argument("--steps", type=int, help="evenly-spaced subset")
    p.add_argument("--commits", nargs="+", help="explicit commit hashes")
    p.add_argument(
        "--rerun", action="store_true", help="benchmark even commits that already have results (e.g. to add benchmarks)"
    )
    p.add_argument("asv_args", nargs="*", help="extra args passed through to `asv run`")
    args = p.parse_args(argv)

    machine = machine_name()
    locks = locked_commits()
    if not locks:
        print("No locks found. Run gen_locks.py first.", file=sys.stderr)
        return 1
    targets = resolve(args, locks)
    print(f"Benchmarking {len(targets)} commit(s) on machine {machine!r}")
    failures = sum(
        not process(commit, lock, python, machine, args.asv_args, rerun=args.rerun) for commit, lock, python in targets
    )
    if failures:
        print(f"\n{failures} commit(s) failed (logged above).", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
