import subprocess
import sys

from benchmarks.bench_import import Import

METADATA = """Metadata-Version: 2.1
Name: gEconpy
Version: 0.0.0
Requires-Dist: packaging>=20
Requires-Dist: not-installed-anywhere>=1
Requires-Dist: colorsys ; extra == "docs"
"""


def test_preload_imports_declared_dependencies_and_skips_the_rest(tmp_path):
    dist_info = tmp_path / "gEconpy-0.0.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(METADATA)
    code, preload = Import().timeraw_gEconpy()
    check = "\nimport sys\nprint('packaging' in sys.modules, 'colorsys' in sys.modules)\n"

    result = subprocess.run(
        [sys.executable, "-I", "-c", f"import sys\nsys.path.insert(0, {str(tmp_path)!r})\n{preload}{check}"],
        capture_output=True,
        text=True,
        check=True,
    )

    assert code == "import gEconpy"
    assert result.stdout.split() == ["True", "False"]


def test_preload_has_no_backslash_for_asv_to_mangle():
    _, preload = Import().timeraw_gEconpy()

    assert "\\" not in preload
