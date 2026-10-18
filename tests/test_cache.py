import pytest

from benchmarks import _cache


class Unpicklable:
    def __reduce__(self):
        raise TypeError("cannot pickle")


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    directory = tmp_path / "cache"
    monkeypatch.setattr(_cache, "CACHE_DIR", directory)
    monkeypatch.setattr(_cache, "_MEMO", {})
    monkeypatch.setattr(_cache, "_gEconpy_version", lambda: "2.0.0")
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
def test_new_gEconpy_version_misses(gcn_file, monkeypatch):
    _cache.load_or_build("warm", gcn_file, lambda: "built by 2.0.0")
    monkeypatch.setattr(_cache, "_gEconpy_version", lambda: "2.0.1")

    assert _cache.load_or_build("warm", gcn_file, lambda: "built by 2.0.1") == "built by 2.0.1"


def test_failed_dump_publishes_no_entry(cache_dir, gcn_file):
    with pytest.raises(TypeError, match="cannot pickle"):
        _cache.load_or_build("warm", gcn_file, Unpicklable)

    assert not list(cache_dir.glob("*.pkl"))
