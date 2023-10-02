from sympy import *
import sympy.printing.c
from sympy.codegen.rewriting import create_expand_pow_optimization

def generate_statement(lhs, rhs, t):

    expand_pow = create_expand_pow_optimization(99)
    expr = sympy.printing.c.ccode(expand_pow(rhs), standard="C99")
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
  const {t} eta1,{third_coord}
  const NekDouble * dofs
){{
  static_assert(false, "Implementation not defined.");
  return {t}(0);
}}
/**
 * TODO
 */
template <size_t NUM_MODES>
inline int flop_count(){{
  static_assert(false, "Implementation not defined.");
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
  const {t} eta1,{third_coord}
  const NekDouble * dofs
){{
{instr_str}
  return {geom_inst.generate_variable()};
}}
"""
        funcs += func

    funcs += f"}} // namespace NESO::GeneratedEvaluation::{namespace}"

    return ops, funcs

