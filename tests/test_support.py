import tracemalloc

import pytest

from benchmarks._support import peak_traced_bytes

ALLOCATION = 20_000_000


def test_peak_traced_bytes_sees_an_allocation_inside_the_call():
    assert peak_traced_bytes(lambda: bytearray(ALLOCATION)) >= ALLOCATION


def test_peak_traced_bytes_stops_tracing_when_the_call_raises():
    def fail():
        raise RuntimeError("build failed")

    with pytest.raises(RuntimeError, match="build failed"):
        peak_traced_bytes(fail)

    assert not tracemalloc.is_tracing()
