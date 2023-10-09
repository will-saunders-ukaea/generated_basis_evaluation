import functools
from sympy import *
from shape_base import *
from ast_base import *


class ProjectBase:
    def _generate(self):
        """
        Generate the expressions required to evaluate the expansion at a point.

        :returns: List of (lhs, rhs) variables and expressions.
        """
        self._dofs.set_evaluate_mode(False)

        qoi, local_stride, local_id = self.generate_variable()
        gmodes = self._get_modes()
        g = []
        for modei, gx in enumerate(gmodes):
            # di_read = self._dofs.generate_variable(modei)
            index_sym = symbols(f"pindex_{modei}")
            index_node = Initialise(
                "int", Assign(index_sym, modei * local_stride + local_id)
            )
            index_node.preserve_ints = True
            index_node.count_ops = False
            g.append(index_node)
            di = self._dofs.array_access(index_sym)
            g.append(Assign(di, di + qoi * gx[1]))

        return g

    def generate_variable(self):
        """
        :returns: The symbol the evaluation will be stored on.
        """
        qoi = symbols(f"qoi")
        local_stride = symbols(f"local_stride")
        local_id = symbols(f"local_id")
        return qoi, local_stride, local_id
