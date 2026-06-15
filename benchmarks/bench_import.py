"""Import cost of gEconpy's own modules.

`setup` evicts gEconpy from the module cache so each timed import re-executes gEconpy's package
initialisation (its heavy third-party dependencies stay cached, so this isolates gEconpy's own
import overhead rather than re-importing pytensor/pymc each time).
"""

import sys


def _evict_gEconpy():
    for name in [k for k in sys.modules if k.startswith("gEconpy")]:
        del sys.modules[name]


class Import:
    def setup(self):
        _evict_gEconpy()

    def teardown(self):
        _evict_gEconpy()

    def time_gEconpy(self):
        import gEconpy  # noqa: F401

    def time_parser(self):
        _evict_gEconpy()
        from gEconpy import parser  # noqa: F401

    def time_model_build(self):
        _evict_gEconpy()
        from gEconpy.model import build  # noqa: F401

    def time_statespace(self):
        _evict_gEconpy()
        from gEconpy.model import statespace  # noqa: F401
