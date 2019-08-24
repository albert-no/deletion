# doublehiddenlp.py
import numpy as np

from deletion_utils.bin_num_util import bin_to_num, num_to_bin
from deletion_utils.mis_util import *

from pulp import *


def hidden_lp(n, verbose=False, binary=False):
    print(f'n = {n}, Formulating LP')
    n_power = np.power(2, n)
    n_minus_power = np.power(2, n-1)
    n_plus_power = np.power(2, n+1)

    model = LpProblem("Hidden LP", LpMaximize)
    if binary:
        V = LpVariable.dicts("V", [i for i in range(n_power)], lowBound=0, cat='Binary')
        fname = f'./lpsols/binary_lp_sol_n{n}.txt'
    else:
        V = LpVariable.dicts("V", [i for i in range(n_power)], lowBound=0)
        fname = f'./lpsols/lp_sol_n{n}.txt'

    model += (pulp.lpSum(V))
    
    hidden_weight = get_hidden_weight_matrix(n)
    for col_idx in range(0, n_minus_power):
        col = hidden_weight[:, col_idx]
        edge_list = np.where(col>0)[0]
        model += (pulp.lpSum([V[idx] for idx in edge_list]) <= 1)

    hidden_weight = get_hidden_weight_matrix(n+1)
    for row_idx in range(0, n_plus_power):
        row = hidden_weight[row_idx, :]
        edge_list = np.where(row>0)[0]
        model += (pulp.lpSum([V[idx] for idx in edge_list]) <= 1)

    print(f'n = {n}, Solving LP')
    model.solve()
    opt = value(model.objective)
    print(f'n={n}, opt={opt}, binary={binary}, status: {LpStatus[model.status]}')
    if verbose:
        cnt = 0
        for v in model.variables():
            print(v.name, '=', v.varValue)
            cnt += 1
        print(f'number of zeros = {cnt}')
    # with open(fname, 'w') as f:
    #     f.write(f'n={n}, opt={opt}\n')
    #     for v in model.variables():
    #         f.write(f'{v.name}, {v.varValue}\n')


if __name__ == "__main__":
    for binary in [False, True]:
        for n in range(4, 10):
            hidden_lp(n, verbose=False, binary=binary)
