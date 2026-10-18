import tracemalloc

import pytest

from benchmarks._support import MODEL_LADDER, gcn_path, peak_traced_bytes
from benchmarks.steady_state.bench_steady_state import NO_STEADY_STATE

ALLOCATION = 20_000_000


def test_peak_traced_bytes_sees_an_allocation_inside_the_call():
    assert peak_traced_bytes(lambda: bytearray(ALLOCATION)) >= ALLOCATION


def test_peak_traced_bytes_stops_tracing_when_the_call_raises():
    def fail():
        raise RuntimeError("build failed")

    with pytest.raises(RuntimeError, match="build failed"):
        peak_traced_bytes(fail)

    assert not tracemalloc.is_tracing()


@pytest.mark.parametrize("stem", [*MODEL_LADDER.values(), *NO_STEADY_STATE.values()])
def test_every_benchmark_fixture_is_vendored(stem):
    assert gcn_path(stem).is_file()
