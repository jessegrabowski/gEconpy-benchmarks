import sys

import pytest

from benchmarks.crossover.bench_core import FIXTURE, _build


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
