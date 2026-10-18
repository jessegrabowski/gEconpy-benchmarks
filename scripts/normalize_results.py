#!/usr/bin/env python
"""Collapse per-commit date-locked environments into one logical environment.

Each historical commit is benchmarked in its own date-locked env, so every result carries a distinct
``env_name`` (its ``.cache/envs/<commit>`` path). asv plots one series per ``env_name``, which turns
the history into hundreds of single-point series, which is unreadable. This rewrites every result file to a
single shared ``env_name`` so asv draws one continuous line per benchmark across commits. Results for
the same commit (e.g. from separate sweeps) are merged. Results from any other environment (a dev env, an
experiment) are skipped and left in place. Run this before ``asv publish``. It is idempotent.
"""

import json
import re

from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / ".asv" / "results"
ENV = "date-locked"
NON_RESULT_FILES = {"machine.json", "benchmarks.json"}

# asv encodes the interpreter path into env_name: .cache/envs/<name>/.pixi/envs becomes _.cache_envs_<name>_.pixi_envs
ENV_DIR_PATTERN = re.compile(r"_\.cache_envs_(?P<name>[^_]+)_\.pixi_envs_")


def is_date_locked(result):
    """Return True if ``result`` ran in its own commit's date-locked env, or was already normalized."""
    if result.get("env_name") == ENV:
        return True
    match = ENV_DIR_PATTERN.search(result.get("env_name", ""))
    return match is not None and result["commit_hash"].startswith(match["name"])


def normalize_machine(machine_dir):
    by_commit = defaultdict(list)
    for path in sorted(machine_dir.glob("*.json")):
        if path.name in NON_RESULT_FILES:
            continue
        data = json.loads(path.read_text())
        if is_date_locked(data):
            by_commit[data["commit_hash"]].append((path, data))

    for commit, items in by_commit.items():
        merged = items[0][1]
        for _, data in items[1:]:
            merged.setdefault("results", {}).update(data.get("results", {}))
            merged.setdefault("durations", {}).update(data.get("durations", {}))
        merged["env_name"] = ENV
        merged["python"] = ENV
        merged["requirements"] = {}
        # asv groups graph series by the full `params` tuple, which carries its own per-commit
        # `python` path, so normalize that too, or the series stay split.
        merged.setdefault("params", {})["python"] = ENV

        out = machine_dir / f"{commit[:8]}-{ENV}.json"
        for path, _ in items:
            if path != out:
                path.unlink()
        out.write_text(json.dumps(merged))
    return len(by_commit)


def main():
    machines = [d for d in RESULTS_DIR.iterdir() if d.is_dir()]
    for machine_dir in machines:
        commit_count = normalize_machine(machine_dir)
        print(f"{machine_dir.name}: collapsed to {commit_count} commits under env '{ENV}'")


if __name__ == "__main__":
    main()
