"""Disk-backed reuse of compiled models, so eval/solve benchmarks don't recompile every run.

asv runs each benchmark in its own subprocess, so an in-memory cache can't span benchmarks and the cache lives on disk.
Entries are content-addressed by the GCN bytes, the gEconpy, pytensor and Python versions, ``CACHE_SCHEMA``, and a
caller-supplied tag, so a reloaded model always matches the code under test: a new commit (new version) misses and
rebuilds. cloudpickle handles the compiled PyTensor functions (a compiled RBC model is ~0.1MB and reloads ~50x faster
than rebuilding).

Compile-measurement benchmarks deliberately bypass this cache, because they build fresh to time the
compilation.
"""

import hashlib
import re
import sys

from importlib import metadata
from pathlib import Path

import cloudpickle

CACHE_DIR = Path(__file__).resolve().parent.parent / ".model_cache"

# Bump when warm_model's builder changes what a cached model contains.
CACHE_SCHEMA = 1

# hatch-vcs marks a dirty tree with a date-only suffix and versioneer with ".dirty". Neither identifies the code.
DIRTY_VERSION = re.compile(r"(\.d\d{8}|\.dirty)$")

_MEMO = {}


def _gEconpy_version():
    import gEconpy

    return getattr(gEconpy, "__version__", "unknown")


def _stack_versions():
    try:
        pytensor = metadata.version("pytensor")
    except metadata.PackageNotFoundError:
        pytensor = "none"
    python = f"{sys.version_info.major}.{sys.version_info.minor}"
    return {"gEconpy": _gEconpy_version(), "pytensor": pytensor, "python": python}


def _key(tag, gcn_path, stack):
    digest = hashlib.sha1(Path(gcn_path).read_bytes()).hexdigest()[:12]
    versions = f"{stack['gEconpy']}__pt{stack['pytensor']}__py{stack['python']}__s{CACHE_SCHEMA}"
    return f"{tag.replace('/', '_')}__{versions}__{digest}"


def _identifies_code(gEconpy_version):
    return gEconpy_version != "unknown" and DIRTY_VERSION.search(gEconpy_version) is None


def load_or_build(tag, gcn_path, builder):
    """Return a cached compiled object for ``tag``+``gcn_path``, building and caching it on a miss.

    ``builder`` is a zero-argument callable invoked only on a cache miss. When the installed gEconpy version doesn't
    identify its code (a dirty tree, or no version at all), the result is kept for this process only.
    """
    stack = _stack_versions()
    key = _key(tag, gcn_path, stack)
    if key in _MEMO:
        return _MEMO[key]

    if not _identifies_code(stack["gEconpy"]):
        _MEMO[key] = builder()
        return _MEMO[key]

    path = CACHE_DIR / f"{key}.pkl"
    if path.exists():
        with path.open("rb") as fd:
            obj = cloudpickle.load(fd)
    else:
        obj = builder()
        CACHE_DIR.mkdir(exist_ok=True)
        tmp = path.with_suffix(".pkl.tmp")
        with tmp.open("wb") as fd:
            cloudpickle.dump(obj, fd)
        tmp.replace(path)  # atomic publish, so a killed run can't leave a half-written cache

    _MEMO[key] = obj
    return obj
