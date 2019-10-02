# mixbinarylp.py
import numpy as np
import random

from deletion_utils.bin_num_util import bin_to_num, num_to_bin
from deletion_utils.mis_util import *
from deletion_utils.clique import *

from pulp import *


def hidden_lp(n, verbose=False):
    print(f'n = {n}, Formulating LP')
    n_power = np.power(2, n)
    n_minus_power = np.power(2, n-1)
    n_plus_power = np.power(2, n+1)

    model = LpProblem("Hidden LP", LpMaximize)

    n_power_list = list(range(n_power))

    # v_bin = get_vt_general(n, [0, n//2])
    # v_bin = [i for i in range(n_power) if compute_vt_sum(num_to_bin(i, n)) in [0, n]]
    # v_bin = [i for i in range(n_power) if compute_vt_sum(num_to_bin(i, n), raw=True) <= n+1 or compute_vt_sum(num_to_bin(i, n), raw=True)>=(n-2)*(n+1)//2]
    # k = 4
    # v_bin = [i for i in range(n_power) if compute_vt_sum(num_to_bin(i, n), raw=True) <= k*(n+1)]
    v_bin = [i for i in range(n_power) if compute_vt_sum(num_to_bin(i, n)) == 0 or compute_vt_sum(num_to_bin(i, n), raw=True)<= n*(n+1)//4]
    # compare = [i for i in range(n_power) if compute_vt_sum(num_to_bin(i, n)) == 0 and compute_vt_sum(num_to_bin(i, n), raw=True) <= k*(n+1)]

    v_else = [i for i in range(n_power) if i not in v_bin]
    # v_else = []
    # v_else = [i for i in range(n_power) if compute_vt_sum(num_to_bin(i, n), raw=True) > k*(n+1) and compute_vt_sum(num_to_bin(i, n), raw=True) < (k+2)*(n+1)]
    Vbin = [i for i in v_bin]
    Velse = [i for i in v_else]

    VBIN = LpVariable.dicts("VBIN", [i for i in v_bin], lowBound=0, cat='Binary')
    VELSE = LpVariable.dicts("VELSE", [i for i in v_else], lowBound=0, upBound=1)

    model += (pulp.lpSum(VBIN)+pulp.lpSum(VELSE))
    
    if 0 in v_bin:
        model += (VBIN[0] == 1)
    else:
        model += (VELSE[0] == 1)

    if n_power-1 in v_bin:
        model += (VBIN[n_power-1] == 1)
    else:
        model += (VELSE[n_power-1] == 1)

    hidden_weight = get_hidden_weight_matrix(n)
    for col_idx in range(0, n_minus_power):
        col = hidden_weight[:, col_idx]
        edge_list = np.where(col>0)[0]
        v_bin_list = [edge for edge in edge_list if edge in v_bin]
        v_else_list = [edge for edge in edge_list if edge in v_else]
        model += ((pulp.lpSum([VBIN[idx] for idx in v_bin_list])+pulp.lpSum([VELSE[idx] for idx in v_else_list])) <= 1)

    hidden_weight = get_hidden_weight_matrix(n+1)
    for row_idx in range(0, n_plus_power):
        row = hidden_weight[row_idx, :]
        edge_list = np.where(row>0)[0]
        v_bin_list = [edge for edge in edge_list if edge in v_bin]
        v_else_list = [edge for edge in edge_list if edge in v_else]
        model += ((pulp.lpSum([VBIN[idx] for idx in v_bin_list])+pulp.lpSum([VELSE[idx] for idx in v_else_list])) <= 1)

    # add clique constraint
    # cliques = findclique(n)
    # for clique in cliques:
    #     clique_bin = [edge for edge in clique if edge in v_bin]
    #     clique_else = [edge for edge in clique if edge in v_else]
    #     model += ((pulp.lpSum([VBIN[idx] for idx in clique_bin])+pulp.lpSum([VELSE[idx] for idx in clique_else])) <= 1)

    print(f'n = {n}, Solving LP')
    model.solve()
    opt = value(model.objective)
    print(f'n={n}, opt={opt}, status: {LpStatus[model.status]}')
    # print(f'target = {len(compare)}')
    print(' ')

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
    for n in range(11, 12):
        hidden_lp(n, verbose=True)
