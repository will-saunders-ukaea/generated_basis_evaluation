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
  const int local_stride,
  const int local_id,
  const {t} eta0,
  const {t} eta1,
  [[maybe_unused]] const {t} eta2,
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
  const int local_stride,
  const int local_id,
  const {t} eta0,
  const {t} eta1,
  [[maybe_unused]] const {t} eta2,
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

    print(proj_sources)
