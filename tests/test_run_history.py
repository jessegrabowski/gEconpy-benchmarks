import pytest
import run_history


@pytest.mark.parametrize(("failing", "exit_code"), [(set(), 0), ({"b" * 40}, 1)], ids=["all-succeed", "one-fails"])
def test_main_exit_code_reports_failures(monkeypatch, failing, exit_code):
    targets = [("a" * 40, None, "3.13"), ("b" * 40, None, "3.13")]
    monkeypatch.setattr(run_history, "machine_name", lambda: "test-machine")
    monkeypatch.setattr(run_history, "locked_commits", lambda: {"aaaaaaaaaa": (None, "3.13")})
    monkeypatch.setattr(run_history, "resolve", lambda _args, _locks: targets)
    monkeypatch.setattr(run_history, "process", lambda commit, *_args, **_kwargs: commit not in failing)

    assert run_history.main([]) == exit_code
