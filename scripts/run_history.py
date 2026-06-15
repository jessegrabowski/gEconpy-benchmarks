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
import platform
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
    import json

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
    """Map commit-prefix -> lock path, from filenames ``<date>_<shorthash>.lock``."""
    out = {}
    for lock in gen_locks.LOCKS_DIR.glob("*.lock"):
        short = lock.stem.split("_", 1)[1]
        out[short] = lock
    return out


def build_env(commit, lock):
    """Reconstruct the date-locked env from its lock; return the env's python path."""
    date = gen_locks.commit_date(commit)
    python = gen_locks.pick_python(commit)
    work = ENVS_CACHE / commit[:10]
    work.mkdir(parents=True, exist_ok=True)
    (work / "pixi.toml").write_text(gen_locks.manifest(date, python))
    (work / "pixi.lock").write_bytes(lock.read_bytes())
    run(["pixi", "install", "--locked", "--manifest-path", str(work / "pixi.toml")])
    py = work / ".pixi" / "envs" / "default" / "bin" / "python"
    if not py.exists():
        raise FileNotFoundError(f"locked env python not found at {py}")
    return py


def install_geconpy(commit, python):
    run(["git", "-C", str(gen_locks.GECONPY_CLONE), "checkout", "--quiet", "--force", commit])
    run([str(python), "-m", "pip", "install", str(gen_locks.GECONPY_CLONE),
         "--no-build-isolation", "--quiet", "--no-deps"])


def benchmark(commit, python, machine, asv_args):
    run(["asv", "run", f"--python={python}", "--set-commit-hash", commit,
         "--machine", machine, "--show-stderr", *asv_args], cwd=str(REPO_ROOT))


def process(commit, lock, machine, asv_args):
    if already_done(commit, machine):
        print(f"  skip   {commit[:10]} (results exist)")
        return True
    try:
        python = build_env(commit, lock)
        install_geconpy(commit, python)
        benchmark(commit, python, machine, asv_args)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"  FAIL   {commit[:10]}: {exc}")
        return False
    print(f"  done   {commit[:10]}")
    return True


def resolve(args, locks):
    if args.commits:
        wanted = [(c, locks[c[:10]]) for c in args.commits if c[:10] in locks]
    else:
        gen_locks.ensure_clone()
        order = gen_locks.run(
            ["git", "-C", str(gen_locks.GECONPY_CLONE), "rev-list", "--reverse", args.range or "main"]
        ).split()
        wanted = [(full, locks[full[:10]]) for full in order if full[:10] in locks]
    if args.steps and args.steps < len(wanted):
        step = len(wanted) / args.steps
        wanted = [wanted[int(i * step)] for i in range(args.steps)]
    return wanted


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--range", help="git rev-list range (default: main)")
    p.add_argument("--steps", type=int, help="evenly-spaced subset")
    p.add_argument("--commits", nargs="+", help="explicit commit hashes")
    p.add_argument("asv_args", nargs="*", help="extra args passed through to `asv run`")
    args = p.parse_args(argv)

    machine = machine_name()
    locks = locked_commits()
    if not locks:
        print("No locks found. Run gen_locks.py first.", file=sys.stderr)
        return 1
    targets = resolve(args, locks)
    print(f"Benchmarking {len(targets)} commit(s) on machine {machine!r}")
    failures = sum(not process(c, lp, machine, args.asv_args) for c, lp in targets)
    if failures:
        print(f"\n{failures} commit(s) failed (logged above).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
