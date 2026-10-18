"""Import cost of gEconpy itself, timed in a fresh interpreter.

asv runs a `timeraw_` benchmark's setup string untimed in a new process, then times the code. The setup imports every
runtime dependency the installed gEconpy declares, so the timed ``import gEconpy`` measures gEconpy's own modules. A
dependency whose import name differs from its distribution name is skipped, and so is one missing from the environment.
"""


class Import:
    def timeraw_gEconpy(self):
        # asv embeds this string in a non-raw triple-quoted string, so it must not contain backslashes.
        preload = (
            "import importlib\n"
            "import importlib.util\n"
            "import itertools\n"
            "from importlib import metadata\n"
            "for requirement in metadata.requires('gEconpy') or []:\n"
            "    if 'extra ==' in requirement:\n"
            "        continue\n"
            "    name = ''.join(itertools.takewhile(lambda ch: ch.isalnum() or ch in '-_.', requirement))\n"
            "    module = name.replace('-', '_').lower()\n"
            "    if module and importlib.util.find_spec(module) is not None:\n"
            "        importlib.import_module(module)\n"
        )
        return "import gEconpy", preload
