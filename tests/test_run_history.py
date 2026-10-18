import argparse

import pytest
import run_history


def commits_args(*commits):
    return argparse.Namespace(commits=list(commits), range=None, steps=None)


def test_resolve_expands_a_short_hash(clone, tmp_path):
    lock = tmp_path / f"2026-01-01_{clone[:10]}_py3.13.lock"
    locks = {clone[:10]: (lock, "3.13")}

    assert run_history.resolve(commits_args(clone[:7]), locks) == [(clone, lock, "3.13")]


def test_resolve_rejects_an_unknown_commit(clone):
    with pytest.raises(ValueError, match="is not a commit"):
        run_history.resolve(commits_args("deadbeef"), {clone[:10]: (None, "3.13")})


@pytest.mark.parametrize(("failing", "exit_code"), [(set(), 0), ({"b" * 40}, 1)], ids=["all-succeed", "one-fails"])
def test_main_exit_code_reports_failures(monkeypatch, failing, exit_code):
    targets = [("a" * 40, None, "3.13"), ("b" * 40, None, "3.13")]
    monkeypatch.setattr(run_history, "machine_name", lambda: "test-machine")
    monkeypatch.setattr(run_history, "locked_commits", lambda: {"aaaaaaaaaa": (None, "3.13")})
    monkeypatch.setattr(run_history, "resolve", lambda _args, _locks: targets)
    monkeypatch.setattr(run_history, "process", lambda commit, *_args, **_kwargs: commit not in failing)

    assert run_history.main([]) == exit_code
