import subprocess

import gen_locks
import pytest


@pytest.fixture
def clone(tmp_path, monkeypatch):
    """Make a one-commit git repo stand in for the gEconpy clone and return that commit's full hash."""
    repo = tmp_path / "gEconpy"
    repo.mkdir()
    git = ["git", "-C", str(repo), "-c", "user.name=test", "-c", "user.email=test@example.com"]
    subprocess.run([*git, "init", "--quiet"], check=True)
    subprocess.run([*git, "commit", "--quiet", "--allow-empty", "-m", "only commit"], check=True)
    monkeypatch.setattr(gen_locks, "GECONPY_CLONE", repo)
    monkeypatch.setattr(gen_locks, "ensure_clone", lambda: None)
    return subprocess.run([*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
