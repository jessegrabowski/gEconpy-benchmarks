from importlib import metadata

import pytest

from benchmarks import _cache

STACK = {"gEconpy": "2.0.0", "pytensor": "3.2.4", "python": "3.13"}


class Unpicklable:
    def __reduce__(self):
        raise TypeError("cannot pickle")


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    directory = tmp_path / "cache"
    monkeypatch.setattr(_cache, "CACHE_DIR", directory)
    monkeypatch.setattr(_cache, "_MEMO", {})
    monkeypatch.setattr(_cache, "_stack_versions", lambda: dict(STACK))
    monkeypatch.setattr(_cache, "_gEconpy_is_editable", lambda: False)
    return directory


@pytest.fixture
def gcn_file(tmp_path):
    path = tmp_path / "model.gcn"
    path.write_text("block HOUSEHOLD {};")
    return path


def test_builder_runs_once_and_disk_hit_skips_it(cache_dir, gcn_file, monkeypatch):
    calls = []

    def builder():
        calls.append(1)
        return {"compiled": True}

    assert _cache.load_or_build("warm", gcn_file, builder) == {"compiled": True}
    monkeypatch.setattr(_cache, "_MEMO", {})
    assert _cache.load_or_build("warm", gcn_file, builder) == {"compiled": True}

    assert len(calls) == 1
    assert len(list(cache_dir.glob("*.pkl"))) == 1


@pytest.mark.usefixtures("cache_dir")
def test_editing_the_gcn_misses(gcn_file):
    _cache.load_or_build("warm", gcn_file, lambda: "first")
    gcn_file.write_text("block HOUSEHOLD { };")

    assert _cache.load_or_build("warm", gcn_file, lambda: "second") == "second"


@pytest.mark.usefixtures("cache_dir")
@pytest.mark.parametrize(
    ("component", "new_version"), [("gEconpy", "2.0.1"), ("pytensor", "3.3.0"), ("python", "3.14")]
)
def test_new_stack_version_misses(gcn_file, monkeypatch, component, new_version):
    _cache.load_or_build("warm", gcn_file, lambda: "old stack")
    monkeypatch.setattr(_cache, "_stack_versions", lambda: {**STACK, component: new_version})

    assert _cache.load_or_build("warm", gcn_file, lambda: "new stack") == "new stack"


@pytest.mark.usefixtures("cache_dir")
def test_schema_bump_misses(gcn_file, monkeypatch):
    _cache.load_or_build("warm", gcn_file, lambda: "old schema")
    monkeypatch.setattr(_cache, "CACHE_SCHEMA", _cache.CACHE_SCHEMA + 1)

    assert _cache.load_or_build("warm", gcn_file, lambda: "new schema") == "new schema"


@pytest.mark.parametrize("version", ["2.2.1.dev3+g9407fda8e.d20260911", "1.3.0+4.g1a2b3c4.dirty", "unknown"])
def test_unidentifiable_version_stays_off_disk(cache_dir, gcn_file, monkeypatch, version):
    monkeypatch.setattr(_cache, "_stack_versions", lambda: {**STACK, "gEconpy": version})
    calls = []

    def builder():
        calls.append(1)
        return "built"

    assert _cache.load_or_build("warm", gcn_file, builder) == "built"
    assert _cache.load_or_build("warm", gcn_file, builder) == "built"

    assert len(calls) == 1
    assert not cache_dir.exists() or not list(cache_dir.iterdir())


def test_failed_dump_publishes_no_entry(cache_dir, gcn_file):
    with pytest.raises(TypeError, match="cannot pickle"):
        _cache.load_or_build("warm", gcn_file, Unpicklable)

    assert not list(cache_dir.glob("*.pkl"))


def test_editable_install_stays_off_disk(cache_dir, gcn_file, monkeypatch):
    monkeypatch.setattr(_cache, "_gEconpy_is_editable", lambda: True)

    assert _cache.load_or_build("warm", gcn_file, lambda: "built") == "built"
    assert not cache_dir.exists() or not list(cache_dir.iterdir())


class FakeDistribution:
    def __init__(self, direct_url):
        self.direct_url = direct_url

    def read_text(self, filename):
        return self.direct_url if filename == "direct_url.json" else None


@pytest.mark.parametrize(
    ("direct_url", "expected"),
    [
        ('{"dir_info": {"editable": true}, "url": "file:///src/gEconpy"}', True),
        ('{"dir_info": {"editable": false}, "url": "file:///src/gEconpy"}', False),
        ('{"dir_info": {}, "url": "file:///src/gEconpy"}', False),
        (None, False),
    ],
    ids=["editable", "editable-false", "local-non-editable", "wheel"],
)
def test_gEconpy_is_editable_reads_direct_url(monkeypatch, direct_url, expected):
    monkeypatch.setattr(metadata, "distribution", lambda _name: FakeDistribution(direct_url))

    assert _cache._gEconpy_is_editable() is expected


def test_gEconpy_is_editable_without_gEconpy_installed(monkeypatch):
    def missing(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "distribution", missing)

    assert _cache._gEconpy_is_editable() is False
