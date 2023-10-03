from sympy import *
import functools
from dof_reader import *
from jacobi import *
from basis_functions import *


class Quadrilateral:

    namespace = "Quadrilateral"
    helper_class = "ExpansionLooping::Quadrilateral"

    def __init__(self, P):

        self.P = P
        self.dofs = DofReader("dofs", self.total_num_modes())
        self.eta0 = symbols("eta0")
        self.eta1 = symbols("eta1")
        jacobi0 = GenJacobi(self.eta0)
        jacobi1 = GenJacobi(self.eta1)
        self.dir0 = eModified_A(P, self.eta0, jacobi0)
        self.dir1 = eModified_A(P, self.eta1, jacobi1)
        self.ndim = 2
        self.common = [jacobi0, self.dir0, jacobi1, self.dir1, self.dofs]

    def total_num_modes(self):
        return self.P * self.P

    def generate_variable(self):
        return symbols(f"eval_{self.eta0}_{self.eta1}")

    def get_blocks(self):
        return self.common + [
            self,
        ]


class QuadrilateralEvaluate(Quadrilateral):
    def generate(self):
        ev = self.generate_variable()
        g = []

        mode = 0
        tmps = []
        for qx in range(self.P):
            for px in range(self.P):
                tmp_mode = symbols(f"eval_{self.eta0}_{self.eta1}_{mode}")
                tmps.append(tmp_mode)
                g.append(
                    (
                        tmp_mode,
                        self.dofs.generate_variable(mode)
                        * self.dir0.generate_variable(px)
                        * self.dir1.generate_variable(qx),
                    )
                )
                mode += 1

        g.append((ev, functools.reduce(lambda x, y: x + y, tmps)))
        return g
