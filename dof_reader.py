from sympy import *
from ast_base import *


class DofReader:
    def __init__(self, sym, max_index):
        # self.sym = MatrixSymbol(sym, 1, max_index)
        self.sym = IndexedBase(sym, shape=(max_index,))
        self.max_index = max_index
        self._g = self._generate()
        self._evaluate_mode = True

    def generate_variable(self, ix):
        assert ix >= 0
        assert ix < self.max_index
        return symbols(f"dof_{ix}")

    def array_access(self, ix):
        return self.sym[ix]

    def set_evaluate_mode(self, mode):
        self._evaluate_mode = mode

    def generate(self):
        if self._evaluate_mode:
            return self._g
        else:
            return []

    def _generate(self):
        g = []
        for ix in range(self.max_index):
            a = Assign(self.generate_variable(ix), self.sym[ix])
            n = Initialise(None, a)
            n.preserve_ints = True
            g.append(n)
        return g
