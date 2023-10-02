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
    geom_inst = geom_type(P)

    instr, ops = generate_block(geom_inst.get_blocks(), t)
    instr_str = "\n".join(["  " + ix for ix in instr])
    
    third_coord = f"\nconst {t} eta2," if geom_inst.ndim == 3 else ""
    func = f"""
template <>
inline {t} quadrilateral_evaluate_scalar<{P}>(
  const {t} eta0,
  const {t} eta1,{third_coord}
  const NekDouble * dofs
){{
{instr_str}
  return {geom_inst.generate_variable()};
}}
    """

    return ops, func

