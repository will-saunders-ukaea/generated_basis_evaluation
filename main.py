import sys
import os
from sympy import *
from basis_functions import *
from generate_source import *
from quadrilateral import *


def header_name(t):
    return f"{t.namespace.lower()}.hpp"


if __name__ == "__main__":

    dir_include = os.path.join(sys.argv[1], "include")
    dir_include_gen_dir = os.path.join(dir_include, "nektar_interface", "expansion_looping", "generated")
    if not os.path.exists(dir_include_gen_dir):
        os.makedirs(dir_include_gen_dir)
 
    P = 8
    
    types = (
        QuadrilateralEvaluate,
    )
    
    header_list = []
    for tx in types:
        filename = header_name(tx)
        header_list.append(f'#include "{filename}"')
        eval_src = generate_vector_wrappers(P, tx)
        with open(os.path.join(dir_include_gen_dir, filename), "w+") as fh:
            fh.write(eval_src)

    header_list = "\n".join(header_list)
    wrapper_header = f"""
#ifndef _NESO_GENERATED_BASIS_EVALUATION_H__
#define _NESO_GENERATED_BASIS_EVALUATION_H__

#include <neso_particles.hpp>
using namespace NESO::Particles;
#include <CL/sycl.hpp>

#include <nektar_interface/expansion_looping/expansion_looping.hpp>

{header_list}

#endif
"""

    wrapper_filename = os.path.join(dir_include_gen_dir, "generated.hpp")
    with open(wrapper_filename, "w+") as fh:
        fh.write(wrapper_header)


