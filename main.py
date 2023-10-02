from sympy import *
from basis_functions import *
from generate_source import *
from quadrilateral import *

if __name__ == "__main__":
    
    P = 4
    quad_eval_ops, quad_eval_src = generate_evaluate(P, QuadrilateralEvaluate, t = "sycl::vec<REAL, VECTOR_LENGTH>")

    print(quad_eval_src)
