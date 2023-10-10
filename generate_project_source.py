from generate_source import *


def generate_project(P, geom_type, t):

    namespace = geom_type.namespace
    shape_name = namespace.lower()
    third_coord = f"\nconst {t} eta2," if geom_type.ndim == 3 else ""

    funcs = f"""namespace NESO::GeneratedProjection::{namespace} {{

/// The minimum number of modes which implementations are generated for.
inline constexpr int mode_min = 2;
/// The maximum number of modes which implementations are generated for.
inline constexpr int mode_max = {P};

/**
 * TODO
 */
template <size_t NUM_MODES>
inline void project(
  const std::size_t local_size,
  const std::size_t local_id,
  const {t} eta0,
  const {t} eta1,
  [[maybe_unused]] const {t} eta2,
  const REAL qoi,
  REAL * dofs
){{
  return;
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
inline void project<{px}>(
  const std::size_t local_size,
  const std::size_t local_id,
  const {t} eta0,
  const {t} eta1,
  [[maybe_unused]] const {t} eta2,
  const REAL qoi,
  REAL * dofs
){{
{instr_str}
  return;
}}
"""
        funcs += func

    funcs += generate_get_flop_count(P)
    funcs += f"}} // namespace NESO::GeneratedProjection::{namespace}"

    return funcs


def generate_project_wrappers(P, geom_type, headers=[]):
    t = "REAL"
    proj_sources = generate_project(P, geom_type, t)
    namespace = geom_type.namespace
    shape_name = namespace.lower()
    proj_type = geom_type.helper_class
    ndim = geom_type.ndim

    cases = []
    for px in range(2, P + 1):
        cases.append(
            f"""
    case {px}:
      project_scalar<{px}>(
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
void project_scalar<{px}>(
    SYCLTargetSharedPtr sycl_target,
    ParticleGroupSharedPtr particle_group, 
    Sym<REAL> sym,
    const int component,
    const int shape_count,
    REAL * k_global_coeffs,
    const int * h_coeffs_offsets,
    const int * h_cells_iterset,
    EventStack &event_stack
  );
"""
        )

    specs = "\n".join(specs)

    funcs = f"""

namespace NESO::GeneratedProjection::{namespace} {{

    template <size_t NUM_MODES>
    void project_scalar(
        SYCLTargetSharedPtr sycl_target,
        ParticleGroupSharedPtr particle_group, 
        Sym<REAL> sym,
        const int component,
        const int shape_count,
        REAL * k_global_coeffs,
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
    REAL * k_global_coeffs,
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

    funcs += f"}} // namespace NESO::GeneratedProjection::{namespace}\n"
    funcs = header_init + proj_sources + funcs
    return funcs


def generate_project_template_specialisation(p, geom_type, t):
    namespace = geom_type.namespace
    shape_name = namespace.lower()
    evaluation_type = geom_type.helper_class
    ndim = geom_type.ndim
    num_dofs = geom_type.total_num_modes(p)
    sub_dict = {
        "NAMESPACE": namespace,
        "EVALUATION_TYPE": evaluation_type,
        "PX": p,
        "NUM_DOFS": num_dofs,
    }

    if ndim == 2:
        mid_kernel = (
            """
            REAL xi0, xi1, xi2, eta0, eta1, eta2;
            xi0 = k_ref_positions[cellx][0][global_id];
            xi1 = k_ref_positions[cellx][1][global_id];
            loop_type.loc_coord_to_loc_collapsed(xi0, xi1, xi2, &eta0, &eta1,
                                                 &eta2);
        """
            % sub_dict
        )
    elif ndim == 3:
        mid_kernel = (
            """
            REAL xi0, xi1, xi2, eta0, eta1, eta2;
            xi0 = k_ref_positions[cellx][0][global_id];
            xi1 = k_ref_positions[cellx][1][global_id];
            xi2 = k_ref_positions[cellx][2][global_id];
            loop_type.loc_coord_to_loc_collapsed(xi0, xi1, xi2, &eta0, &eta1,
                                                 &eta2);
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
namespace NESO::GeneratedProjection::%(NAMESPACE)s {

template <>
void project_scalar<%(PX)s>(
    SYCLTargetSharedPtr sycl_target,
    ParticleGroupSharedPtr particle_group, 
    Sym<REAL> sym,
    const int component,
    const int shape_count,
    REAL * k_global_coeffs,
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

  auto k_qoi = (*particle_group)[sym]->cell_dat.device_ptr();
  const int k_component = component;

  const size_t local_size = get_num_local_work_items(
    sycl_target,
    static_cast<size_t>(%(NUM_DOFS)s) * sizeof(REAL),
    128);

  for(int cell_idx=0 ; cell_idx<cells_iterset_size ; cell_idx++){
    const int cellx = h_cells_iterset[cell_idx];
    const int dof_offset = h_coeffs_offsets[cellx];
    REAL *dofs = k_global_coeffs + dof_offset;
    const int num_particles = mpi_rank_dat->h_npart_cell[cellx]; 
    const size_t global_size = get_global_size((std::size_t) num_particles, local_size);

    auto event_loop = sycl_target->queue.submit([&](sycl::handler &cgh) {

      sycl::accessor<REAL, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          local_mem(
            sycl::range<1>(local_size * static_cast<size_t>(%(NUM_DOFS)s)), 
            cgh
          );

      cgh.parallel_for<>(
          sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(local_size)),
          [=](sycl::nd_item<1> nd_idx) {
            const size_t global_id = nd_idx.get_global_linear_id();
            const size_t local_id = nd_idx.get_local_linear_id();
            
            for(int dx=0 ; dx<(%(NUM_DOFS)s) ; dx++){
                local_mem[local_id + dx*local_size] = 0.0;
            }

            if (global_id < num_particles){
              const REAL qoi = k_qoi[cellx][k_component][global_id];
              %(EVALUATION_TYPE)s loop_type{};
              %(MID_KERNEL)s
              project<%(PX)s>(local_size, local_id, eta0, eta1, eta2, qoi, &local_mem[0]);
            }

            sycl::group_barrier(nd_idx.get_group());
            
            if (local_id == 0){
                for(int dx=0 ; dx<(%(NUM_DOFS)s) ; dx++){
                    REAL tmp_dof = 0.0;
                    for(int ix=0 ; ix<local_size ; ix++){
                        tmp_dof += local_mem[ix + dx*local_size];
                    }
                    sycl::atomic_ref<REAL, sycl::memory_order::relaxed,
                    sycl::memory_scope::device>
                    coeff_atomic_ref(dofs[dx]);
                    coeff_atomic_ref.fetch_add(tmp_dof);
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
    funcs += f"}} // namespace NESO::GeneratedProjection::{namespace}"

    return funcs




def generate_project_sources(P, geom_type):
    t = "REAL"
    sources = []
    for px in range(2, P + 1):
        src = generate_project_template_specialisation(px, geom_type, t)
        sources.append((px, src))
    return sources
