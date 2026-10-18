from pathlib import Path

from asv_runner.discovery import disc_benchmarks

BENCHMARK_ROOT = str(Path(__file__).resolve().parents[1] / "benchmarks")


def test_peakmem_only_in_crossover():
    """Keep peakmem to Crossover, because asv's peakmem also counts memory used during setup."""
    peakmem = [
        benchmark.name
        for benchmark in disc_benchmarks(BENCHMARK_ROOT)
        if benchmark.name.split(".")[-1].startswith("peakmem_")
    ]

    assert peakmem == ["crossover.bench_core.Crossover.peakmem_build_solve"]
