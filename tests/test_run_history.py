import argparse
import subprocess

import gen_locks
import pytest
import run_history


def commits_args(*commits):
    return argparse.Namespace(commits=list(commits), range=None, steps=None)


def test_resolve_expands_a_short_hash(clone, tmp_path):
    lock = tmp_path / f"2026-01-01_{clone[:10]}_py3.13.lock"
    locks = {clone[:10]: (lock, "3.13")}

    assert run_history.resolve(commits_args(clone[:7]), locks) == ([(clone, lock, "3.13")], [])


def test_resolve_rejects_an_unknown_commit(clone):
    with pytest.raises(ValueError, match="is not a commit"):
        run_history.resolve(commits_args("deadbeef"), {clone[:10]: (None, "3.13")})


def test_main_fails_a_requested_commit_without_a_lock(clone, monkeypatch, capsys):
    monkeypatch.setattr(run_history, "machine_name", lambda: "test-machine")
    monkeypatch.setattr(run_history, "locked_commits", lambda: {"aaaaaaaaaa": (None, "3.13")})

    assert run_history.main(["--commits", clone[:7]]) == 1
    assert f"FAIL   {clone[:10]} (no lock; run gen_locks.py --commits {clone[:10]} first)" in capsys.readouterr().out


def test_main_still_benchmarks_locked_commits_alongside_an_unlocked_one(clone, monkeypatch):
    git = ["git", "-C", str(gen_locks.GECONPY_CLONE), "-c", "user.name=test", "-c", "user.email=test@example.com"]
    subprocess.run([*git, "commit", "--quiet", "--allow-empty", "-m", "second commit"], check=True)
    unlocked = subprocess.run([*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    benchmarked = []
    monkeypatch.setattr(run_history, "machine_name", lambda: "test-machine")
    monkeypatch.setattr(run_history, "locked_commits", lambda: {clone[:10]: (None, "3.13")})
    monkeypatch.setattr(run_history, "process", lambda commit, *_args, **_kwargs: benchmarked.append(commit) or True)

    assert run_history.main(["--commits", clone[:7], unlocked[:7]]) == 1
    assert benchmarked == [clone]


@pytest.mark.parametrize(("failing", "exit_code"), [(set(), 0), ({"b" * 40}, 1)], ids=["all-succeed", "one-fails"])
def test_main_exit_code_reports_failures(monkeypatch, failing, exit_code):
    targets = [("a" * 40, None, "3.13"), ("b" * 40, None, "3.13")]
    monkeypatch.setattr(run_history, "machine_name", lambda: "test-machine")
    monkeypatch.setattr(run_history, "locked_commits", lambda: {"aaaaaaaaaa": (None, "3.13")})
    monkeypatch.setattr(run_history, "resolve", lambda _args, _locks: (targets, []))
    monkeypatch.setattr(run_history, "process", lambda commit, *_args, **_kwargs: commit not in failing)

    assert run_history.main([]) == exit_code


@pytest.mark.parametrize("stored", [True, False], ids=["stored-manifest", "rebuilt-manifest"])
def test_locked_manifest_prefers_the_stored_one(tmp_path, monkeypatch, stored):
    lock = tmp_path / "2026-01-01_aaaaaaaaaa_py3.13.lock"
    if stored:
        gen_locks.manifest_path(lock).write_text("stored manifest")
    monkeypatch.setattr(gen_locks, "commit_date", lambda _commit: "2026-01-01")
    monkeypatch.setattr(gen_locks, "full_deps", lambda _commit: {"numpy"})

    expected = "stored manifest" if stored else gen_locks.manifest("2026-01-01", "3.13", {"numpy"})
    assert run_history.locked_manifest("a" * 40, lock, "3.13") == expected
