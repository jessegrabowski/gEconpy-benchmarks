"""GCN parsing: the pure-grammar pass versus the full load-to-primitives pass."""

from .._support import gcn_path

LADDER = ["small", "rbc", "medium", "large"]


class Parse:
    params = [LADDER]
    param_names = ["model"]

    def setup(self, model):
        from gEconpy.parser import load_gcn_file, parse_gcn

        self._parse_gcn = parse_gcn
        self._load_gcn_file = load_gcn_file
        self.path = gcn_path(model)
        self.text = self.path.read_text()

    def time_grammar(self, model):
        """Lex + parse the GCN text into a parse tree (no block simplification)."""
        self._parse_gcn(self.text)

    def time_load_to_primitives(self, model):
        """Full parse + block simplification down to model primitives."""
        self._load_gcn_file(self.path)
