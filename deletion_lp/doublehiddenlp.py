# doublehiddenlp.py
import numpy as np

from deletion_utils.bin_num_util import bin_to_num, num_to_bin
from deletion_utils.mis_util import *
from deletion_utils.clique import *

from pulp import *


def hidden_lp(n, verbose=False, binary=False):
    print(f'n = {n}, Formulating LP')
    n_power = np.power(2, n)
    n_minus_power = np.power(2, n-1)
    n_plus_power = np.power(2, n+1)

    model = LpProblem("Hidden LP", LpMaximize)
    if binary:
        V = LpVariable.dicts("V", [i for i in range(n_power)], lowBound=0, cat='Binary')
    else:
        V = LpVariable.dicts("V", [i for i in range(n_power)], lowBound=0)

    model += (pulp.lpSum(V))
    
    model += (V[0] == 1)
    model += (V[n_power-1] == 1)

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

    # # add 3-cliques contraints
    # cliques = findclique(n)
    # for clique in cliques:
    #     model += (pulp.lpSum([V[idx] for idx in clique]) <= 1)

    print(f'n = {n}, Solving LP')
    model.solve()
    opt = value(model.objective)
    print(f'n={n}, opt={opt}, binary={binary}, status: {LpStatus[model.status]}')

    if verbose:
        cnt = 0
        for v in model.variables():
            if v.varValue > 0:
                bin_str = num_to_bin(int(v.name[2:]), n)
                vt_sum = compute_vt_sum(bin_str, raw=True)
                print(f'{v.name} = {v.varValue}, VT_sum = {vt_sum}')
                cnt += 1
        print(f'number of nonzeros = {cnt}')


if __name__ == "__main__":
    for binary in [False]:
        for n in range(6, 12):
            hidden_lp(n, verbose=False, binary=binary)
