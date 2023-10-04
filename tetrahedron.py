from sympy import *
import functools
from dof_reader import *
from jacobi import *
from basis_functions import *
from evaluate_base import *


class Tetrahedron(ShapeBase):
    """
    Base class for Tetrahedron implementations.
    """

    namespace = "Tetrahedron"
    helper_class = "ExpansionLooping::Tetrahedron"
    ndim = 3

    def __init__(self, P):
        """
        Create a class to operate with expansions of a certain order over a
        Tetrahedron.

        :param P: The number of modes used by an expansion over this geometry
                  object.
        """

        """Number of modes in the expansion."""
        self.P = P
        self._dofs = DofReader("dofs", self.total_num_modes())
        self._eta0 = symbols("eta0")
        self._eta1 = symbols("eta1")
        self._eta2 = symbols("eta2")
        jacobi0 = GenJacobi(self._eta0)
        jacobi1 = GenJacobi(self._eta1)
        jacobi2 = GenJacobi(self._eta2)
        self._dir0 = eModified_A(P, self._eta0, jacobi0)
        self._dir1 = eModified_B(P, self._eta1, jacobi1)
        self._dir2 = eModified_C(P, self._eta2, jacobi2)
        self._common = [
            jacobi0,
            self._dir0,
            jacobi1,
            self._dir1,
            jacobi2,
            self._dir2,
            self._dofs,
        ]
        self._g = self._generate()

    def total_num_modes(self):
        """
        :returns: The total number of modes (DOFs) touched by this expansion
        looping.
        """
        mode = 0
        for px in range(self.P):
            for qx in range(self.P - px):
                for rx in range(self.P - px - qx):
                    mode += 1
        return mode

    def _get_modes(self):
        """
        :returns: The (lhs, rhs) tuples that evaluate each mode.
        """

        g = []
        mode = 0
        for px in range(self.P):
            for qx in range(self.P - px):
                for rx in range(self.P - px - qx):
                    tmp_mode = symbols(
                        f"eval_{self._eta0}_{self._eta1}_{self._eta2}_{mode}"
                    )

                    c0 = self._dir0.generate_variable(px)
                    c1 = self._dir1.generate_variable(px, qx)
                    c2 = self._dir2.generate_variable(px, qx, rx)

                    if mode == 1:
                        rhs = c2
                    elif (px == 0) and (qx == 1):
                        rhs = c1 * c2
                    else:
                        rhs = c0 * c1 * c2

                    g.append((tmp_mode, rhs))
                    mode += 1

        return g


class TetrahedronEvaluate(Tetrahedron, EvaluateBase):
    """
    Implementation to evaluate expansions over a Tetrahedron.
    """

    pass