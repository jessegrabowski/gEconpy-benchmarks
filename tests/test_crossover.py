import sys

from pathlib import Path

import pytest

from asv_runner.discovery import disc_benchmarks

from benchmarks.crossover.bench_core import FIXTURE, _build

BENCHMARK_ROOT = str(Path(__file__).resolve().parents[1] / "benchmarks")

# Recorded history is keyed on these. A change means the edit dropped every Crossover result on the dashboard.
CROSSOVER_VERSIONS = {
    "crossover.bench_core.Crossover.peakmem_build_solve": "05bbe6df74e2",
    "crossover.bench_core.Crossover.time_build": "46967eecac4a",
    "crossover.bench_core.Crossover.time_build_solve": "a555b3d265ab",
    "crossover.bench_core.Crossover.time_solve": "9fa2eaf0dd18",
    "crossover.bench_core.Crossover.time_steady_state": "6e2c6ef94e7b",
}


@pytest.fixture
def fake_gEconpy(tmp_path, monkeypatch):
    """Put an empty ``gEconpy`` package first on ``sys.path`` and return its directory for the test to populate."""
    package = tmp_path / "gEconpy"
    package.mkdir()
    (package / "__init__.py").write_text("")
    for name in [name for name in sys.modules if name == "gEconpy" or name.startswith("gEconpy.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(tmp_path))
    return package


def test_build_falls_back_to_1x_without_gEconpy_model(fake_gEconpy):
    (fake_gEconpy / "classes").mkdir()
    (fake_gEconpy / "classes" / "__init__.py").write_text(
        "class gEconModel:\n    def __init__(self, path, verbose):\n        self.path = path\n"
    )

    model = _build(FIXTURE)

    assert type(model).__name__ == "gEconModel"
    assert model.path == str(FIXTURE)


def test_build_reraises_a_missing_dependency_inside_2x(fake_gEconpy):
    (fake_gEconpy / "model").mkdir()
    (fake_gEconpy / "model" / "__init__.py").write_text("")
    (fake_gEconpy / "model" / "build.py").write_text("import some_missing_dependency\n")

    with pytest.raises(ModuleNotFoundError, match="some_missing_dependency"):
        _build(FIXTURE)


def test_crossover_versions_are_unchanged():
    versions = {
        benchmark.name: benchmark.version[:12]
        for benchmark in disc_benchmarks(BENCHMARK_ROOT)
        if benchmark.name.startswith("crossover.")
    }

    assert versions == CROSSOVER_VERSIONS
