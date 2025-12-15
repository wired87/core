"""
math engine
get lis all datatypes
get list all operators

meine intuition ist der hammer aber ich zerdenke alles

Rules
EQ Graph:
id = key




"""


import itertools

# WICHTIG FÜR MATH G ENGINE Alle relevanten Python-Operatoren als Funktionen
OPS = {
    '+': "add",
    '-': "sub",
    '*': "mul",
    '/': "div",
    '**': "pow",
    '%': "mod",
    '//': "floordiv",
    '==': "eq",
    '!=': "neq",
    '>': "gt",
    '<': "lt",
    '>=': "ge",
    '<=': "le",
    '&': "and",
    '|': "or",
    '^': "xor",
    '<<': "lshift",
    '>>': "rshift",
    '(': "lparen",
    ')': "rparen",
    '@': "matmul",
    '=': "assign"
}


def apply_all_operator_combinations(variables):
    results = {}

    keys = list(variables.keys())
    combos = itertools.combinations(keys, 2)
    op_patterns = list(OPS.items())

    for (a, b) in combos:
        for symbol, func in op_patterns:
            name = f"{a}{symbol}{b}"
            try:
                res = func(variables[a], variables[b])
                results[name] = res
            except Exception as e:
                results[name] = f"Error: {e}"

    return results



