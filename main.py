import sys
import os
from sympy import *
from basis_functions import *
from generate_source import *
from quadrilateral import *

if __name__ == "__main__":

    dir_include = os.path.join(sys.argv[1], "include")
    dir_include_gen_dir = os.path.join(dir_include, "nektar_interface", "expansion_looping", "generated")
    if not os.path.exists(dir_include_gen_dir):
        os.makedirs(dir_include_gen_dir)
 
    P = 8
    quad_eval_src = generate_vector_wrappers(P, QuadrilateralEvaluate)
    with open(os.path.join(dir_include_gen_dir, "quadrilateral.hpp"), "w+") as fh:
        fh.write(quad_eval_src)
    print(quad_eval_src)
    """

    z = symbols("z")
    j = GenJacobi(z)
    j(2,1,1)

    s, _ = generate_block((j,))
    print("\n".join(s))
    """






