from sympy import *
import functools
from dof_reader import *
from jacobi import *
from basis_functions import *
from evaluate_base import *
from project_base import *


class Pyramid(ShapeBase):
    """
    Base class for Pyramid implementations.
    """

    namespace = "Pyramid"
    helper_class = "ExpansionLooping::Pyramid"
    ndim = 3

    def __init__(self, P):
        """
        Create a class to operate with expansions of a certain order over a
        Pyramid.

        :param P: The number of modes used by an expansion over this geometry
                  object.
        """

        """Number of modes in the expansion."""
        self.P = P
        self._dofs = DofReader("dofs", self.total_num_modes(self.P))
        self._eta0 = symbols("eta0")
        self._eta1 = symbols("eta1")
        self._eta2 = symbols("eta2")
        jacobi0 = GenJacobi(self._eta0)
        jacobi1 = GenJacobi(self._eta1)
        jacobi2 = GenJacobi(self._eta2)
        self._dir0 = eModified_A(P, self._eta0, jacobi0)
        self._dir1 = eModified_A(P, self._eta1, jacobi1)
        self._dir2 = eModified_PyrC(P, self._eta2, jacobi2)
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

    @staticmethod
    def total_num_modes(P):
        """
        :param P: Number of modes in one dimension.
        :returns: The total number of modes (DOFs) touched by this expansion
        looping.
        """
        mode = 0
        for px in range(P):
            for qx in range(P):
                for rx in range(P - max(px, qx)):
                    mode += 1
        return mode

    def _get_modes(self):
        """
        :returns: The (lhs, rhs) tuples that evaluate each mode.
        """

        g = []
        mode = 0
        for px in range(self.P):
            for qx in range(self.P):
                for rx in range(self.P - max(px, qx)):
                    tmp_mode = symbols(
                        f"eval_{self._eta0}_{self._eta1}_{self._eta2}_{mode}"
                    )

                    c0 = self._dir0.generate_variable(px)
                    c1 = self._dir1.generate_variable(qx)
                    c2 = self._dir2.generate_variable(px, qx, rx)

                    if mode == 1:
                        rhs = c2
                    else:
                        rhs = c0 * c1 * c2

                    g.append((tmp_mode, rhs))
                    mode += 1

        return g


class PyramidEvaluate(Pyramid, EvaluateBase):
    """
    Implementation to evaluate expansions over a Pyramid.
    """

    pass


class PyramidProject(Pyramid, ProjectBase):
    """
    Implementation to project expansions over a Pyramid.
    """

    pass
