from sympy import *
import sympy.printing.c
from sympy.codegen.rewriting import create_expand_pow_optimization
from sympy.codegen.ast import *
from sympy.printing.c import C99CodePrinter
from functools import cmp_to_key, total_ordering
from ast_base import *


class C99RemoveIntLiterals(C99CodePrinter):
    """
    This should not be needed but the expand_pow optimisation refuses to work
    properly with remove_integer.
    """

    def _print_Integer(self, flt):
        return f"{flt}.0"


def contains_indexed(expr):
    def walk(e):
        c = False
        if issubclass(type(e), Indexed):
            c = True
        args = e.args
        for ax in args:
            c = c or walk(ax)
        return c

    ci = walk(expr)
    return ci


def generate_statement(node, t):

    lhs = node[0]
    rhs = node[1]
    preserve_ints = node.preserve_ints
    expand_pow = create_expand_pow_optimization(99)

    rhs = expand_pow(rhs)
    ci = contains_indexed(rhs) or node.preserve_ints
    if ci:
        c_printer = C99CodePrinter()
    else:
        c_printer = C99RemoveIntLiterals()

    expr = c_printer.doprint(rhs)

    if type(node) is Assign:
        stmt = f"{lhs} = {expr};"
    elif type(node) is Initialise:
        if node.type is not None:
            t = node.type
        stmt = f"{t} {lhs}({expr});"
    else:
        raise RuntimeError("Unknown node type.")

    return stmt


def sort_exprs(rewrite_list, ast_cse_list):
    # assume the output_list and cse_list are each individually sorted.

    rr = [rx for rx in rewrite_list]
    v = [ox[0] for ox in rewrite_list]
    to_insert = [r for r in ast_cse_list]

    max_dep_index = 0
    while len(to_insert) > 0:
        cx = to_insert.pop(0)
        catoms = cx[1].atoms()
        for ax in catoms:
            if ax in v:
                max_dep_index = max(max_dep_index, v.index(ax) + 1)
        v.insert(max_dep_index, cx[0])
        rr.insert(max_dep_index, cx)
        max_dep_index += 1

    return rr


symbols_generator = numbered_symbols()

def pass_cse(expr_list, t):
    output_steps = [lx[0] for lx in expr_list]
    steps = [lx[1] for lx in expr_list]

    cse_list = cse(steps, symbols=symbols_generator, optimizations="basic")

    ast_cse_list = to_ast(t, cse_list[0])
    rewrite_list = cse_list[1]
    for ri, rx in enumerate(rewrite_list):
        expr_list[ri][1] = rx
    sorted_exprs = sort_exprs(expr_list, ast_cse_list)
    return sorted_exprs


def generate_block(components, t="REAL"):

    g = []
    for cx in components:
        cx_tmp = cx.generate()
        cx_tmp = to_ast(t, cx_tmp)
        g.append(cx_tmp)

    instr = []
    ops = 0
    for gx in g:
        sorted_exprs = pass_cse(gx, t)
        for node in sorted_exprs:
            e = generate_statement(node, t)
            if node.count_ops:
                ops += node[1].count_ops()
            instr.append(e)

    return instr, ops


def generate_get_flop_count(P):

    cases = []
    for px in range(2, P + 1):
        cases.append(
            f"""
        case {px}:
            return flop_count<{px}>();
            break;
        """
        )
    cases = "\n".join(cases)

    func = f"""
/**
 * Get the FLOP count for a single evaluation for a single particle for a given
 * number of modes. 
 * 
 * @param p Number of modes.
 * @returns FLOP count. -1 if the number of requested modes is outside the
 * generated range.
 */
inline int get_flop_count(const int p){{
    switch (p) {{
{cases}
        default:
            return -1;
    }}
}}
"""
    return func


def generate_evaluate(P, geom_type, t, constructor_assigns=False):

    namespace = geom_type.namespace
    shape_name = namespace.lower()
    third_coord = f"\nconst {t} eta2," if geom_type.ndim == 3 else ""

    funcs = f"""namespace NESO::GeneratedEvaluation::{namespace} {{

/// The minimum number of modes which implementations are generated for.
inline constexpr int mode_min = 2;
/// The maximum number of modes which implementations are generated for.
inline constexpr int mode_max = {P};

/**
 * TODO
 */
template <size_t NUM_MODES>
inline {t} evaluate(
  const {t} eta0,
  const {t} eta1,
  [[maybe_unused]] const {t} eta2,
  const REAL * dofs
){{
  return {t}(0);
}}
/**
 * TODO
 */
template <size_t NUM_MODES>
inline int flop_count(){{
  return (0);
}}
    """

    for px in range(2, P + 1):

        geom_inst = geom_type(px)
        instr, ops = generate_block(geom_inst.get_blocks(), t)
        instr_str = "\n".join(["  " + ix for ix in instr])

        func = f"""
/**
 * TODO
 */
template <>
inline int flop_count<{px}>(){{
  return {ops};
}}
/**
 * TODO
 * Sympy flop count: {ops}
 */
template <>
inline {t} evaluate<{px}>(
  const {t} eta0,
  const {t} eta1,
  [[maybe_unused]] const {t} eta2,
  const REAL * dofs
){{
{instr_str}
  return {geom_inst.generate_variable()};
}}
"""
        funcs += func

    funcs += generate_get_flop_count(P)
    funcs += f"}} // namespace NESO::GeneratedEvaluation::{namespace}"

    return funcs


def generate_template_specialisation(p, geom_type, t = "sycl::vec<REAL, NESO_VECTOR_LENGTH>"):
    namespace = geom_type.namespace
    shape_name = namespace.lower()
    evaluation_type = geom_type.helper_class
    ndim = geom_type.ndim
    sub_dict = {
        "NAMESPACE": namespace,
        "EVALUATION_TYPE": evaluation_type,
        "PX": p,
    }

    if ndim == 2:
        mid_kernel = (
            """
            REAL eta0_local[NESO_VECTOR_LENGTH];
            REAL eta1_local[NESO_VECTOR_LENGTH];
            REAL eval_local[NESO_VECTOR_LENGTH];
            for(int ix=0 ; ix<NESO_VECTOR_LENGTH ; ix++){
              eta0_local[ix] = 0.0;
              eta1_local[ix] = 0.0;
              eval_local[ix] = 0.0;
            }
            int cx = 0;
            for(int ix=layer_start ; ix<layer_end ; ix++){
              REAL xi0, xi1, xi2, eta0, eta1, eta2;
              xi0 = k_ref_positions[cellx][0][ix];
              xi1 = k_ref_positions[cellx][1][ix];
              loop_type.loc_coord_to_loc_collapsed(xi0, xi1, xi2, &eta0, &eta1,
                                                   &eta2);
              eta0_local[cx] = eta0;
              eta1_local[cx] = eta1;
              cx++;
            }

            sycl::local_ptr<const REAL> eta0_ptr(eta0_local);
            sycl::local_ptr<const REAL> eta1_ptr(eta1_local);

            sycl::vec<REAL, NESO_VECTOR_LENGTH> eta0, eta1, eta2;
            eta0.load(0, eta0_ptr);
            eta1.load(0, eta1_ptr);

            const sycl::vec<REAL, NESO_VECTOR_LENGTH> eval = 
              evaluate<%(PX)s>(eta0, eta1, eta2, dofs);
        """
            % sub_dict
        )
    elif ndim == 3:
        mid_kernel = (
            """
            REAL eta0_local[NESO_VECTOR_LENGTH];
            REAL eta1_local[NESO_VECTOR_LENGTH];
            REAL eta2_local[NESO_VECTOR_LENGTH];
            REAL eval_local[NESO_VECTOR_LENGTH];
            for(int ix=0 ; ix<NESO_VECTOR_LENGTH ; ix++){
              eta0_local[ix] = 0.0;
              eta1_local[ix] = 0.0;
              eta2_local[ix] = 0.0;
              eval_local[ix] = 0.0;
            }
            int cx = 0;
            for(int ix=layer_start ; ix<layer_end ; ix++){
              REAL xi0, xi1, xi2, eta0, eta1, eta2;
              xi0 = k_ref_positions[cellx][0][ix];
              xi1 = k_ref_positions[cellx][1][ix];
              xi2 = k_ref_positions[cellx][2][ix];
              loop_type.loc_coord_to_loc_collapsed(xi0, xi1, xi2, &eta0, &eta1,
                                                   &eta2);
              eta0_local[cx] = eta0;
              eta1_local[cx] = eta1;
              eta2_local[cx] = eta2;
              cx++;
            }

            sycl::local_ptr<const REAL> eta0_ptr(eta0_local);
            sycl::local_ptr<const REAL> eta1_ptr(eta1_local);
            sycl::local_ptr<const REAL> eta2_ptr(eta2_local);

            sycl::vec<REAL, NESO_VECTOR_LENGTH> eta0, eta1, eta2;
            eta0.load(0, eta0_ptr);
            eta1.load(0, eta1_ptr);
            eta2.load(0, eta2_ptr);

            const sycl::vec<REAL, NESO_VECTOR_LENGTH> eval = 
              evaluate<%(PX)s>(eta0, eta1, eta2, dofs);
        """
            % sub_dict
        )
    else:
        raise RuntimeError("Dimension not implemented")

    sub_dict.update(
        {
            "MID_KERNEL": mid_kernel,
        }
    )

    funcs = (
        """
namespace NESO::GeneratedEvaluation::%(NAMESPACE)s {

template <>
void evaluate_vector<%(PX)s>(
    SYCLTargetSharedPtr sycl_target,
    ParticleGroupSharedPtr particle_group, 
    Sym<REAL> sym,
    const int component,
    const int shape_count,
    const REAL * k_global_coeffs,
    const int * h_coeffs_offsets,
    const int * h_cells_iterset,
    EventStack &event_stack
  ) {

  const int cells_iterset_size = shape_count;
  if (cells_iterset_size == 0) {
    return;
  }

  auto mpi_rank_dat = particle_group->mpi_rank_dat;

  const auto k_ref_positions =
      (*particle_group)[Sym<REAL>("NESO_REFERENCE_POSITIONS")]
          ->cell_dat.device_ptr();

  auto k_output = (*particle_group)[sym]->cell_dat.device_ptr();
  const int k_component = component;

  for(int cell_idx=0 ; cell_idx<cells_iterset_size ; cell_idx++){
    const int cellx = h_cells_iterset[cell_idx];
    const int dof_offset = h_coeffs_offsets[cellx];
    const REAL *dofs = k_global_coeffs + dof_offset;

    const int num_particles = mpi_rank_dat->h_npart_cell[cellx]; 

    const auto div_mod =
        std::div(static_cast<long long>(num_particles), static_cast<long long>(NESO_VECTOR_LENGTH));
    const std::size_t num_blocks =
        static_cast<std::size_t>(div_mod.quot + (div_mod.rem == 0 ? 0 : 1));

    const size_t ls = 128;
    const size_t gs = get_global_size((std::size_t) num_blocks, ls);

    auto event_loop = sycl_target->queue.submit([&](sycl::handler &cgh) {

      cgh.parallel_for<>(
          //sycl::range<1>(static_cast<size_t>(num_blocks)),
          //[=](sycl::id<1> idx) {
          sycl::nd_range<1>(sycl::range<1>(gs), sycl::range<1>(ls)),
          [=](sycl::nd_item<1> nd_idx) {
            const size_t idx = nd_idx.get_global_linear_id();
            if (idx < num_blocks){
              const INT layer_start = idx * NESO_VECTOR_LENGTH;
              const INT layer_end = std::min(INT(layer_start + NESO_VECTOR_LENGTH), INT(num_particles));
              %(EVALUATION_TYPE)s loop_type{};
              %(MID_KERNEL)s
              sycl::local_ptr<REAL> eval_ptr(eval_local);
              eval.store(0, eval_ptr);

              cx = 0;
              for(int ix=layer_start ; ix<layer_end ; ix++){
                k_output[cellx][k_component][ix] = eval_local[cx];
                cx++;
              }
            }
          });
    });

    event_stack.push(event_loop);

  }
  return;
}
"""
        % sub_dict
    )
    funcs += f"}} // namespace NESO::GeneratedEvaluation::{namespace}"

    return funcs


def generate_vector_sources(P, geom_type):
    t = "sycl::vec<REAL, NESO_VECTOR_LENGTH>"
    sources = []
    for px in range(2, P + 1):
        src = generate_template_specialisation(px, geom_type, t)
        sources.append((px, src))
    return sources


def generate_vector_wrappers(P, geom_type, headers=[]):
    t = "sycl::vec<REAL, NESO_VECTOR_LENGTH>"
    eval_sources = generate_evaluate(P, geom_type, t, True)
    namespace = geom_type.namespace
    shape_name = namespace.lower()
    evaluation_type = geom_type.helper_class
    ndim = geom_type.ndim

    cases = []
    for px in range(2, P + 1):
        cases.append(
            f"""
    case {px}:
      evaluate_vector<{px}>(
        sycl_target,
        particle_group, 
        sym,
        component,
        shape_count,
        k_global_coeffs,
        h_coeffs_offsets,
        h_cells_iterset,
        event_stack
      );
      return true;"""
        )

    cases = "\n".join(cases)

    specs = []
    for px in range(2, P + 1):
        specs.append(
            f"""
template <>
void evaluate_vector<{px}>(
    SYCLTargetSharedPtr sycl_target,
    ParticleGroupSharedPtr particle_group, 
    Sym<REAL> sym,
    const int component,
    const int shape_count,
    const REAL * k_global_coeffs,
    const int * h_coeffs_offsets,
    const int * h_cells_iterset,
    EventStack &event_stack
  );
"""
        )

    specs = "\n".join(specs)

    funcs = f"""

namespace NESO::GeneratedEvaluation::{namespace} {{

    template <size_t NUM_MODES>
    void evaluate_vector(
        SYCLTargetSharedPtr sycl_target,
        ParticleGroupSharedPtr particle_group, 
        Sym<REAL> sym,
        const int component,
        const int shape_count,
        const REAL * k_global_coeffs,
        const int * h_coeffs_offsets,
        const int * h_cells_iterset,
        EventStack &event_stack
      );
    
    {specs}

    """

    funcs += f"""
inline bool generated_call_exists(
    const int num_modes,
    SYCLTargetSharedPtr sycl_target,
    ParticleGroupSharedPtr particle_group, 
    Sym<REAL> sym,
    const int component,
    const int shape_count,
    const REAL * k_global_coeffs,
    const int * h_coeffs_offsets,
    const int * h_cells_iterset,
    EventStack &event_stack
  ) {{
    
    if (shape_count == 0){{
        return false;
    }}

  if (
      (num_modes >= 2) && (num_modes <= {P})
  ) {{
    switch(num_modes) {{
{cases}

      default:
        return false;
    }}
  }} else {{
    return false;
  }}
}}
"""
    header_init = (
        "\n".join(["#include " + hx for hx in headers])
        + """
#include <neso_particles.hpp>
using namespace NESO::Particles;
#include <CL/sycl.hpp>
#include <type_traits>
#include <SpatialDomains/MeshGraph.h>
using namespace Nektar::LibUtilities;
    """
    )

    funcs += f"}} // namespace NESO::GeneratedEvaluation::{namespace}\n"
    funcs = header_init + eval_sources + funcs
    return funcs
