#!/usr/bin/env python
"""Collapse per-commit date-locked environments into one logical environment.

Each historical commit is benchmarked in its own date-locked env, so every result carries a distinct ``env_name``
(its ``.cache/envs/<commit prefix>`` path). asv plots one series per ``env_name``, which turns the history into hundreds
of single-point series. This merges each commit's date-locked results into one file under a shared ``env_name``, so asv
draws one continuous line per benchmark across commits. When two runs of a commit measured the same benchmark, the
newer measurement is kept. Results from any other environment (a dev env, an experiment) are left out.

Without ``--in-place`` it only reports what it would do. ``--in-place`` rewrites the results directory: it writes the
merged files and deletes every input it merged or left out, so the directory holds only what ``asv publish`` should
render. CI runs it that way on a throwaway checkout.
"""

import argparse
import copy
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


def merge_commit(results):
    """Merge one commit's date-locked results, keeping the most recent measurement of each benchmark."""
    layouts = {tuple(result["result_columns"]) for result in results}
    if len(layouts) > 1:
        raise ValueError(f"commit {results[0]['commit_hash'][:10]} has results with different column layouts")
    started_at = results[0]["result_columns"].index("started_at")

    newest = {}
    for result in results:
        for name, entry in result["results"].items():
            if name not in newest or entry[started_at] > newest[name][started_at]:
                newest[name] = entry

    def last_start(result):
        return max((entry[started_at] for entry in result["results"].values()), default=0)

    by_recency = sorted(results, key=last_start)
    merged = copy.deepcopy(by_recency[-1])
    merged["results"] = newest
    merged["durations"] = {key: value for result in by_recency for key, value in result.get("durations", {}).items()}
    merged["env_name"] = ENV
    merged["python"] = ENV
    merged["requirements"] = {}
    # asv groups graph series by the full `params` tuple, which carries its own per-commit `python` path.
    merged.setdefault("params", {})["python"] = ENV
    return merged


def output_path(machine_dir, commit):
    return machine_dir / f"{commit[:8]}-{ENV}.json"


def normalize_machine(machine_dir, *, in_place):
    """Merge one machine's date-locked results and, if ``in_place``, rewrite the directory to hold only those.

    Returns
    -------
    commit_count : int
        Number of commits with date-locked results.
    excluded : list of Path
        Result files left out because they came from another environment.
    """
    by_commit = defaultdict(list)
    excluded = []
    for path in sorted(machine_dir.glob("*.json")):
        if path.name in NON_RESULT_FILES:
            continue
        result = json.loads(path.read_text())
        if is_date_locked(result):
            by_commit[result["commit_hash"]].append((path, result))
        else:
            excluded.append(path)

    merged = {commit: merge_commit([result for _, result in items]) for commit, items in by_commit.items()}
    if not in_place:
        return len(merged), excluded

    outputs = {output_path(machine_dir, commit): result for commit, result in merged.items()}
    for path, result in outputs.items():
        path.write_text(json.dumps(result))
    inputs = [path for items in by_commit.values() for path, _ in items] + excluded
    for path in inputs:
        if path not in outputs:
            path.unlink()
    return len(merged), excluded


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in-place", action="store_true", help="rewrite the results directory (destructive)")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help="asv results directory")
    args = parser.parse_args(argv)

    for machine_dir in sorted(path for path in args.results_dir.iterdir() if path.is_dir()):
        commit_count, excluded = normalize_machine(machine_dir, in_place=args.in_place)
        verb = "collapsed" if args.in_place else "would collapse"
        print(f"{machine_dir.name}: {verb} {commit_count} commits under env '{ENV}'")
        for path in excluded:
            print(f"  {'removed' if args.in_place else 'would remove'} {path.name} (not a date-locked env)")


if __name__ == "__main__":
    main()
