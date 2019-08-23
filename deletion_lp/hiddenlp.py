# hiddenlp.py
import numpy as np

from deletion_utils.bin_num_util import bin_to_num, num_to_bin

from pulp import *


def hidden_lp(n, verbose=False, binary=False):
    print(f'n = {n}, Formulating LP')
    n_power = np.power(2, n)
    n_minus_power = np.power(2, n-1)

    model = LpProblem("Hidden LP", LpMaximize)
    if binary:
        V = LpVariable.dicts("V", [i for i in range(n_power)], lowBound=0, cat='Binary')
        fname = f'./lpsols/binary_lp_sol_n{n}.txt'
    else:
        V = LpVariable.dicts("V", [i for i in range(n_power)], lowBound=0)
        fname = f'./lpsols/lp_sol_n{n}.txt'

    model += (pulp.lpSum(V))
    
    for x in range(0, n_minus_power):
        bin_str = num_to_bin(x, n-1)
        edge_list = []
        for idx in range(n):
            for bit in ['0', '1']:
                bin_added = bin_str[:idx] + bit + bin_str[idx:]
                num_added = bin_to_num(bin_added)
                if num_added not in edge_list:
                    edge_list.append(num_added)
        model += (pulp.lpSum([V[idx] for idx in edge_list]) <= 1)
    print(f'n = {n}, Solving LP')
    model.solve()
    opt = value(model.objective)
    print(f'n={n}, opt={opt}, status: {LpStatus[model.status]}')
    if verbose:
        cnt = 0
        for v in model.variables():
            print(v.name, '=', v.varValue)
            cnt += 1
        print(f'number of zeros = {cnt}')
    with open(fname, 'w') as f:
        f.write(f'n={n}, opt={opt}\n')
        for v in model.variables():
            f.write(f'{v.name}, {v.varValue}\n')


if __name__ == "__main__":
    for n in range(10, 13):
        hidden_lp(n, verbose=False, binary=True)
