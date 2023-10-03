from sympy import *

def remove_integer(expr):
    subs = {}
    def walk(e):
        if e.is_Integer:
            subs[e] = Float(e)
        for ax in e.args:
            walk(ax)
    walk(expr)
    return expr.subs(subs)
