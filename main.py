import sys
import os
from sympy import *
from basis_functions import *
from generate_source import *
from quadrilateral import *


def header_name_evaluate(t):
    return f"evaluate_{t.namespace.lower()}.hpp"


if __name__ == "__main__":

    dir_include = os.path.join(sys.argv[1], "include")
    dir_include_gen_dir = os.path.join(
        dir_include, "nektar_interface", "expansion_looping", "generated"
    )
    dir_include_cmake_dir = os.path.join(
        "${INC_DIR}", "nektar_interface", "expansion_looping", "generated"
    )
    if not os.path.exists(dir_include_gen_dir):
        os.makedirs(dir_include_gen_dir)

    P = 8

    types = (QuadrilateralEvaluate,)

    header_list = []
    cmake_include_list = []

    for tx in types:
        filename = header_name_evaluate(tx)
        header_list.append(f'#include "{filename}"')
        cmake_include_list.append(os.path.join(dir_include_cmake_dir, filename))

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

    wrapper_filename = os.path.join(dir_include_gen_dir, "generated_evaluate.hpp")
    with open(wrapper_filename, "w+") as fh:
        fh.write(wrapper_header)

    dir_cmake = os.path.join(sys.argv[1], "cmake")
    if not os.path.exists(dir_cmake):
        os.makedirs(dir_cmake)
    dir_cmake_filename = os.path.join(dir_cmake, "GeneratedEvaluate.cmake")

    cmake_include_list = "\n".join(["    " + cx for cx in cmake_include_list])

    cmake_source = f"""# This is a generated file do not edit this file.
# Instead modify the Python code which generates this file.
# This function adds the generated files to evaluate fields at points.

function(ADD_GENERATED_EVALUATION_INCLUDES)

  list(APPEND HEADER_FILES 
    ${{INC_DIR}}/{os.path.relpath(wrapper_filename, dir_include)}
{cmake_include_list}
  )
  set (HEADER_FILES ${{HEADER_FILES}} PARENT_SCOPE)

endfunction()
"""

    with open(dir_cmake_filename, "w+") as fh:
        fh.write(cmake_source)
