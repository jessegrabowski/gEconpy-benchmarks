import gen_locks
import pytest

PYPROJECT_2X = """
[build-system]
requires = ["hatchling", "hatch-vcs"]

[project]
requires-python = ">=3.11"
dependencies = ["numpy>=1.24", "pymc_extras[jax]>=0.1", "scipy ; python_version<'3.14'"]
"""

SETUP_CFG_1X = """
[options]
install_requires =
    numba>=0.55
    sympy
"""

SETUP_PY_EARLY = """
setup(name="gEconpy", install_requires=["numpy", 'pandas>=1.3'])
"""


def use_files(monkeypatch, files):
    monkeypatch.setattr(gen_locks, "_show", lambda _commit, path: files.get(path, ""))


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("numpy>=1.24", "numpy"),
        ("pymc_extras[jax]>=0.1", "pymc-extras"),
        ("scipy ; python_version<'3.14'", "scipy"),
        ("'Sympy==1.12'", "sympy"),
        ("# a comment", ""),
        ("", ""),
    ],
)
def test_norm_dep_returns_conda_name(spec, expected):
    assert gen_locks._norm_dep(spec) == expected


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        ({"pyproject.toml": PYPROJECT_2X}, {"numpy", "pymc-extras", "scipy"}),
        ({"setup.cfg": SETUP_CFG_1X}, {"numba", "sympy"}),
        ({"setup.py": SETUP_PY_EARLY}, {"numpy", "pandas"}),
        ({"pyproject.toml": PYPROJECT_2X, "setup.cfg": SETUP_CFG_1X}, {"numpy", "pymc-extras", "scipy"}),
        ({}, set()),
    ],
    ids=["pyproject", "setup-cfg", "setup-py", "pyproject-wins", "none"],
)
def test_commit_deps_reads_each_era(monkeypatch, files, expected):
    use_files(monkeypatch, files)
    assert gen_locks.commit_deps("abc") == expected


@pytest.mark.parametrize(
    ("files", "expected"),
    [({"pyproject.toml": PYPROJECT_2X}, {"hatchling", "hatch-vcs"}), ({"setup.py": SETUP_PY_EARLY}, {"setuptools"})],
    ids=["build-system", "fallback"],
)
def test_build_deps(monkeypatch, files, expected):
    use_files(monkeypatch, files)
    assert gen_locks.build_deps("abc") == expected


@pytest.mark.parametrize(
    ("requires_python", "date", "expected"),
    [
        (">=3.10, <3.13", "2025-03-15", ["3.12", "3.11", "3.10"]),
        (">=3.11", "2024-01-01", ["3.12", "3.11"]),
        (None, "2021-09-19", ["3.9"]),
        (None, "2021-10-04", ["3.10", "3.9"]),
    ],
    ids=["upper-bound", "floor", "date-cap", "release-day"],
)
def test_candidate_pythons_respects_spec_and_date(monkeypatch, requires_python, date, expected):
    project = "" if requires_python is None else f'[project]\nrequires-python = "{requires_python}"\n'
    use_files(monkeypatch, {"pyproject.toml": project})
    monkeypatch.setattr(gen_locks, "commit_date", lambda _commit: date)
    assert gen_locks.candidate_pythons("abc") == expected
