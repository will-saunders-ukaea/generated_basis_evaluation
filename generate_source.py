from sympy import *
import sympy.printing.c
from sympy.codegen.rewriting import create_expand_pow_optimization
from sympy.codegen.ast import *
from sympy.printing.c import C99CodePrinter



class C99RemoveIntLiterals(C99CodePrinter):
    """
    This should not be needed but the expand_pow optimisation refuses to work
    properly with remove_integer.
    """
    def _print_Integer(self, flt):
        return f"{flt}.0"

def generate_statement(lhs, rhs, t):

    expand_pow = create_expand_pow_optimization(99)
    c_printer = C99RemoveIntLiterals()
    expr = c_printer.doprint(expand_pow(rhs))
    stmt = f"{t} {lhs}({expr});"
    return stmt


def generate_block(components, t="REAL"):
    g = []
    for cx in components:
        g.append(cx.generate())

    instr = []
    symbols_generator = numbered_symbols()

    ops = 0
    for gx in g:
        output_steps = [lx[0] for lx in gx]
        steps = [lx[1] for lx in gx]
        cse_list = cse(steps, symbols=symbols_generator, optimizations="basic")
        for cse_expr in cse_list[0]:
            lhs = cse_expr[0]
            ops += cse_expr[1].count_ops()
            e = generate_statement(lhs, cse_expr[1], t)
            instr.append(e)
        for lhs_v, rhs_v in zip(output_steps, cse_list[1]):
            e = generate_statement(lhs_v, rhs_v, t)
            ops += rhs_v.count_ops()
            instr.append(e)
    
    return instr, ops


def generate_evaluate(P, geom_type, t):

    geom_inst0 = geom_type(0)
    namespace = geom_inst0.namespace
    shape_name = namespace.lower()
    third_coord = f"\nconst {t} eta2," if geom_inst0.ndim == 3 else ""

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

    for px in range(2, P+1):

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

    funcs += f"}} // namespace NESO::GeneratedEvaluation::{namespace}"

    return funcs


def generate_vector_wrappers(P, geom_type):
    t = "sycl::vec<REAL, NESO_VECTOR_LENGTH>"
    eval_sources = generate_evaluate(P, geom_type, t)
    geom_inst0 = geom_type(0)
    namespace = geom_inst0.namespace
    shape_name = namespace.lower()
    evaluation_type = geom_inst0.helper_class

    cases = []
    for px in range(2, P+1):
        cases.append(f"""
    case {px}:
      evaluate_vector<{px}>(
        sycl_target,
        particle_group, 
        sym,
        component,
        map_shape_to_count,
        k_global_coeffs,
        h_coeffs_offsets,
        h_cells_iterset,
        event_stack
      );
      return true;""")
    
    cases = "\n".join(cases)

    funcs = """
#include <type_traits>
namespace NESO::GeneratedEvaluation::%(NAMESPACE)s {


template <size_t NUM_MODES>
inline void evaluate_vector(
    SYCLTargetSharedPtr sycl_target,
    ParticleGroupSharedPtr particle_group, 
    Sym<REAL> sym,
    const int component,
    std::map<ShapeType, int> &map_shape_to_count,
    const REAL * k_global_coeffs,
    const int * h_coeffs_offsets,
    const int * h_cells_iterset,
    EventStack &event_stack
  ) {

  %(EVALUATION_TYPE)s evaluation_type{};
  const ShapeType shape_type = evaluation_type.get_shape_type();
  const int cells_iterset_size = map_shape_to_count.at(shape_type);
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

    auto event_loop = sycl_target->queue.submit([&](sycl::handler &cgh) {
      const int num_particles = mpi_rank_dat->h_npart_cell[cellx]; 

      const auto div_mod =
          std::div(static_cast<long long>(num_particles), static_cast<long long>(NESO_VECTOR_LENGTH));
      const std::size_t num_blocks =
          static_cast<std::size_t>(div_mod.quot + (div_mod.rem == 0 ? 0 : 1));

      cgh.parallel_for<>(
          sycl::range<1>(static_cast<size_t>(num_blocks)),
          [=](sycl::id<1> idx) {

            const INT layer_start = idx * NESO_VECTOR_LENGTH;
            const INT layer_end = std::min(INT(layer_start + NESO_VECTOR_LENGTH), INT(num_particles));
            %(EVALUATION_TYPE)s loop_type{};
            const int k_ndim = loop_type.get_ndim();

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
              if (k_ndim > 1) {
                xi1 = k_ref_positions[cellx][1][ix];
              }
              if (k_ndim > 2) {
                xi2 = k_ref_positions[cellx][2][ix];
              }
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
              evaluate<NUM_MODES>(eta0, eta1, eta2, dofs);

            sycl::local_ptr<REAL> eval_ptr(eval_local);
            eval.store(0, eval_ptr);

            cx = 0;
            for(int ix=layer_start ; ix<layer_end ; ix++){
              k_output[cellx][k_component][ix] = eval_local[cx];
              cx++;
            }
          });
    });

    event_stack.push(event_loop);

  }
  return;
}
""" % {"NAMESPACE": namespace, "EVALUATION_TYPE": evaluation_type}

    funcs += f"""
template <typename COMPONENT_TYPE>
inline bool vector_call_exists(
    const int num_modes,
    SYCLTargetSharedPtr sycl_target,
    ParticleGroupSharedPtr particle_group, 
    Sym<COMPONENT_TYPE> sym,
    const int component,
    std::map<ShapeType, int> &map_shape_to_count,
    const REAL * k_global_coeffs,
    const int * h_coeffs_offsets,
    const int * h_cells_iterset,
    EventStack &event_stack
  ) {{

    if (!(std::is_same_v<COMPONENT_TYPE, REAL> == true)){{
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

    funcs += f"}} // namespace NESO::GeneratedEvaluation::{namespace}\n"
    funcs = eval_sources + funcs
    return funcs



















