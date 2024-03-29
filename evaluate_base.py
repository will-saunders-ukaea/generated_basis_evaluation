import functools
from sympy import *
from shape_base import *


class EvaluateBase:
    def _generate(self):
        """
        Generate the expressions required to evaluate the expansion at a point.

        :returns: List of (lhs, rhs) variables and expressions.
        """

        ev = self.generate_variable()
        gmodes = self._get_modes()
        g = []
        for modei, gx in enumerate(gmodes):
            g.append((gx[0], self._dofs.generate_variable(modei) * gx[1]))

        g.append((ev, functools.reduce(lambda x, y: x + y, [tx[0] for tx in g])))
        return g

    def generate_variable(self):
        """
        :returns: The symbol the evaluation will be stored on.
        """
        if self.ndim == 2:
            return symbols(f"eval_{self._eta0}_{self._eta1}")
        elif self.ndim == 3:
            return symbols(f"eval_{self._eta0}_{self._eta1}_{self._eta2}")
        else:
            raise RuntimeError("Bad number of dimensions.")
