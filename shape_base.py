import functools
from sympy import *


class ShapeBase:
    def get_blocks(self):
        """
        :returns: List of objects which define a generate method.
        """
        return self._common + [
            self,
        ]

    def generate(self):
        """
        :returns: The expressions generated by this class.
        """
        return self._g