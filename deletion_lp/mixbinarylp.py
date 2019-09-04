# mibinarylp.py
import numpy as np
import random

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

    n_power_list = list(range(n_power))
    # vt0 = random.sample(n_power_list, 3*n+1)

    # vt0 = get_vt_general(n, [0, n//2])
    vt0 = [i for i in range(n_power) if compute_vt_sum(num_to_bin(i, n), raw=True) <= (n+1)]
    vt_else = [i for i in range(n_power) if i not in vt0]

    VT0 = LpVariable.dicts("VT0", [i for i in vt0], lowBound=0, cat='Binary')
    VTELSE = LpVariable.dicts("VTELSE", [i for i in vt_else], lowBound=0)

    model += (pulp.lpSum(VT0)+pulp.lpSum(VTELSE))
    
    if 0 in vt0:
        model += (VT0[0] == 1)
    else:
        model += (VTELSE[0] == 1)

    if n_power-1 in vt0:
        model += (VT0[n_power-1] == 1)
    else:
        model += (VTELSE[n_power-1] == 1)
    
    # if n in vt0:
    #     model += (VT0[n] == 0)
    # else:
    #     model += (VTELSE[n] == 0)

    hidden_weight = get_hidden_weight_matrix(n)
    for col_idx in range(0, n_minus_power):
        col = hidden_weight[:, col_idx]
        edge_list = np.where(col>0)[0]
        vt0_list = [edge for edge in edge_list if edge in vt0]
        vt_else_list = [edge for edge in edge_list if edge in vt_else]
        model += ((pulp.lpSum([VT0[idx] for idx in vt0_list])+pulp.lpSum([VTELSE[idx] for idx in vt_else_list])) <= 1)

    hidden_weight = get_hidden_weight_matrix(n+1)
    for row_idx in range(0, n_plus_power):
        row = hidden_weight[row_idx, :]
        edge_list = np.where(row>0)[0]
        vt0_list = [edge for edge in edge_list if edge in vt0]
        vt_else_list = [edge for edge in edge_list if edge in vt_else]
        model += ((pulp.lpSum([VT0[idx] for idx in vt0_list])+pulp.lpSum([VTELSE[idx] for idx in vt_else_list])) <= 1)

    print(f'n = {n}, Solving LP')
    model.solve()
    opt = value(model.objective)
    print(f'n={n}, opt={opt}, binary={binary}, status: {LpStatus[model.status]}')

    if verbose:
        cnt = 0
        for v in model.variables():
            if v.varValue > 0:
                idx = v.name.find('_')
                bin_str = num_to_bin(int(v.name[idx+1:]), n)
                vt_sum = compute_vt_sum(bin_str)
                print(f'{v.name} = {v.varValue}, VT_sum = {vt_sum}')
                cnt += 1
        print(f'number of nonzeros = {cnt}')


if __name__ == "__main__":
    for binary in [False]:
        for n in range(5, 11):
            hidden_lp(n, verbose=False, binary=binary)
