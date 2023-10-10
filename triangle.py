from sympy import *
import functools
from dof_reader import *
from jacobi import *
from basis_functions import *
from evaluate_base import *
from project_base import *

class Triangle(ShapeBase):
    """
    Base class for Triangle implementations.
    """

    namespace = "Triangle"
    helper_class = "ExpansionLooping::Triangle"
    ndim = 2

    def __init__(self, P):
        """
        Create a class to operate with expansions of a certain order over a
        Triangle.

        :param P: The number of modes used by an expansion over this geometry
                  object.
        """
        """Number of modes in the expansion."""
        self.P = P
        self._dofs = DofReader("dofs", self.total_num_modes(self.P))
        self._eta0 = symbols("eta0")
        self._eta1 = symbols("eta1")
        jacobi0 = GenJacobi(self._eta0)
        jacobi1 = GenJacobi(self._eta1)
        self._dir0 = eModified_A(P, self._eta0, jacobi0)
        self._dir1 = eModified_B(P, self._eta1, jacobi1)
        self._common = [jacobi0, self._dir0, jacobi1, self._dir1, self._dofs]
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
            for qx in range(P - px):
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
                tmp_mode = symbols(f"eval_{self._eta0}_{self._eta1}_{mode}")

                d1 = self._dir1.generate_variable(px, qx)
                if mode == 1:
                    rhs = d1
                else:
                    rhs = d1 * self._dir0.generate_variable(px)

                g.append((tmp_mode, rhs))
                mode += 1

        return g


class TriangleEvaluate(Triangle, EvaluateBase):
    """
    Implementation to evaluate expansions over a Triangle.
    """

    pass

class TriangleProject(Triangle, ProjectBase):
    """
    Implementation to project expansions over a Triangle.
    """

    pass
