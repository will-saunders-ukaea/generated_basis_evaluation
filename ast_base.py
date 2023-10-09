from sympy import *


class Node:
    def __init__(self):
        self.preserve_ints = False
        self.count_ops = True


class Assign(Node):
    def __init__(self, lhs, rhs):
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs

    def __getitem__(self, key):
        if key == 0:
            return self.lhs
        elif key == 1:
            return self.rhs
        else:
            raise RuntimeError("Bad key")

    def __setitem__(self, key, value):
        if key == 0:
            self.lhs = value
        elif key == 1:
            self.rhs = value
        else:
            raise RuntimeError("Bad key")


class Initialise(Node):
    def __init__(self, t, assign):
        super().__init__()
        self.type = t
        self.assign = assign

    def __getitem__(self, key):
        return self.assign[key]

    def __setitem__(self, key, value):
        self.assign[key] = value


def to_ast(t, instr_list):
    instr_cast = []
    for ix in instr_list:
        if type(ix) not in (Initialise, Assign):
            ix = Initialise(t, Assign(ix[0], ix[1]))
        instr_cast.append(ix)
    return instr_cast


class constant_name:
    idx = -1
    base_name = "const_factor_"

    def __init__(self):
        self.d = {}

    def get(self, v):
        if v in self.d.keys():
            return self.d[v]
        else:
            self.idx += 1
            n = self.base_name + str(self.idx)
            self.d[v] = n
            return n


def extract_constants(expr):

    cn = constant_name()
    consts = []

    def walk(e):
        args = e.args
        if len(args) > 0:
            new_args = [walk(ax) for ax in args]
            f = e.func(*new_args)
        else:
            if e.is_Number:
                rhs = e.coeff
                lhs = symbols(cn.get(str(e)))
                a = Assign(lhs, RealNumber(rhs))
                consts.append(Initialise("REAL", a))
                f = Symbol(lhs)
            else:
                f = e
        return f

    new_expr = walk(expr)

    return consts, new_expr
