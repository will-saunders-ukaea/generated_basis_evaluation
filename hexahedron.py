from sympy import *
import functools
from dof_reader import *
from jacobi import *
from basis_functions import *
from evaluate_base import *
from project_base import *


class Hexahedron(ShapeBase):
    """
    Base class for Hexahedron implementations.
    """

    namespace = "Hexahedron"
    helper_class = "ExpansionLooping::Hexahedron"
    ndim = 3

    def __init__(self, P):
        """
        Create a class to operate with expansions of a certain order over a
        Hexahedron.

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
        self._dir1 = eModified_A(P, self._eta1, jacobi1)
        self._dir2 = eModified_A(P, self._eta2, jacobi2)
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
        return self.P * self.P * self.P

    def _get_modes(self):
        """
        :returns: The (lhs, rhs) tuples that evaluate each mode.
        """
        g = []
        mode = 0
        for rx in range(self.P):
            for qx in range(self.P):
                for px in range(self.P):
                    tmp_mode = symbols(
                        f"eval_{self._eta0}_{self._eta1}_{self._eta2}_{mode}"
                    )
                    g.append(
                        (
                            tmp_mode,
                            self._dir0.generate_variable(px)
                            * self._dir1.generate_variable(qx)
                            * self._dir2.generate_variable(rx),
                        )
                    )
                    mode += 1
        return g


class HexahedronEvaluate(Hexahedron, EvaluateBase):
    """
    Implementation to evaluate expansions over a Hexahedron.
    """

    pass


class HexahedronProject(Hexahedron, ProjectBase):
    """
    Implementation to project expansions over a Hexahedron.
    """

    pass
