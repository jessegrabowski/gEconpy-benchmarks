import json

import normalize_results
import pytest

COLUMNS = ["result", "params", "version", "started_at", "duration"]
COMMIT = "9407fda8e8a1b2c3d4e5f60718293a4b5c6d7e8f"
LOCKED_ENV = "existing-py_Users_me_gEconpy-benchmarks_.cache_envs_9407fda8e8_.pixi_envs_default_bin_python"


def write_result(directory, filename, env_name, benchmarks, commit=COMMIT):
    """Write an asv result file whose ``benchmarks`` map a name to ``(value, started_at)``."""
    results = {name: [[value], [], "v1", started_at, 1.0] for name, (value, started_at) in benchmarks.items()}
    payload = {
        "commit_hash": commit,
        "env_name": env_name,
        "python": "/some/python",
        "params": {"python": "/some/python"},
        "requirements": {"numpy": "1.0"},
        "result_columns": COLUMNS,
        "results": results,
        "durations": {},
    }
    path = directory / filename
    path.write_text(json.dumps(payload))
    return path


def read_merged(directory):
    return json.loads((directory / f"{COMMIT[:8]}-date-locked.json").read_text())


@pytest.mark.parametrize(
    ("env_name", "expected"),
    [
        (LOCKED_ENV, True),
        (LOCKED_ENV.replace("9407fda8e8", "9407fda"), True),
        ("date-locked", True),
        (LOCKED_ENV.replace("9407fda8e8", "newstack"), False),
        (LOCKED_ENV.replace("9407fda8e8", "05cc74b328"), False),
        ("existing-py_Users_me_mambaforge_envs_geconpy-dev_bin_python", False),
    ],
    ids=["own-env", "own-env-7-char", "normalized", "experiment-env", "other-commit-env", "dev-env"],
)
def test_is_date_locked_requires_the_commit_own_env(env_name, expected):
    assert normalize_results.is_date_locked({"env_name": env_name, "commit_hash": COMMIT}) is expected


def test_merges_locked_runs_and_skips_other_envs(tmp_path):
    write_result(tmp_path, "a-locked.json", LOCKED_ENV, {"bench.time_a": (1.0, 100)})
    write_result(tmp_path, "b-locked.json", LOCKED_ENV, {"bench.time_b": (2.0, 200)})
    dev = write_result(tmp_path, "c-dev.json", "existing-py_mambaforge_envs_dev", {"bench.time_a": (9.0, 300)})

    assert normalize_results.normalize_machine(tmp_path) == 1

    merged = read_merged(tmp_path)
    assert {name: entry[0] for name, entry in merged["results"].items()} == {
        "bench.time_a": [1.0],
        "bench.time_b": [2.0],
    }
    assert merged["env_name"] == merged["python"] == merged["params"]["python"] == "date-locked"
    assert dev.exists()


def test_newest_measurement_wins(tmp_path):
    write_result(tmp_path, "a-rerun.json", LOCKED_ENV, {"bench.time_a": (3.0, 500)})
    write_result(tmp_path, "b-first.json", LOCKED_ENV, {"bench.time_a": (1.0, 100)})

    normalize_results.normalize_machine(tmp_path)

    assert read_merged(tmp_path)["results"]["bench.time_a"][0] == [3.0]


def test_mismatched_column_layouts_raise(tmp_path):
    write_result(tmp_path, "a-locked.json", LOCKED_ENV, {"bench.time_a": (1.0, 100)})
    path = write_result(tmp_path, "b-locked.json", LOCKED_ENV, {"bench.time_b": (2.0, 200)})
    payload = json.loads(path.read_text())
    payload["result_columns"] = [*COLUMNS, "samples"]
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="different column layouts"):
        normalize_results.normalize_machine(tmp_path)
