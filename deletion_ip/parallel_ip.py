# parallel_ip.py
import numpy as np
import random
import time

from deletion_utils.bin_num_util import bin_to_num, num_to_bin
from deletion_utils.mis_util import *
from deletion_utils.clique import *
from deletion_lp.node_from_parallel import NODES

from pulp import *


def parallel_ip_with_vt_bound(n, vt_bound, inc_plus=False, verbose=False):
    print(f'n = {n}, Formulating LP')
    n_power = np.power(2, n)
    n_minus_power = np.power(2, n-1)
    n_vt_sum_array = get_vt_sum_array(n)
    n_minus_vt_sum_array = get_vt_sum_array(n-1)

    if inc_plus:
        n_plus_power = np.power(2, n+1)
        n_plus_vt_sum_array = get_vt_sum_array(n+1)

    model = LpProblem("Parallel IP", LpMaximize)

    n_idx = n_vt_sum_array <= vt_bound
    v_list = [i for i in range(n_power) if n_idx[i]]
    V = LpVariable.dicts("V", v_list, lowBound=0, cat='Binary')

    model += pulp.lpSum(V)

    # n_minus constraints
    hidden_weight = get_hidden_weight_matrix(n)
    n_minus_idx = n_minus_vt_sum_array <= vt_bound
    col_list = [i for i in range(n_minus_power) if n_minus_idx[i]]
    for col_idx in col_list:
        col = hidden_weight[:, col_idx]
        edge_list = np.where(col>0)[0]
        model += (pulp.lpSum([V[idx] for idx in edge_list if n_idx[idx]])<=1)

    # n_plus constraints
    # additional constraints might reduce the running time but it seems not
    if inc_plus:
        hidden_weight = get_hidden_weight_matrix(n+1)
        n_plus_idx = (n_plus_vt_sum_array <= (vt_bound+n+1))
        row_list = [i for i in range(n_plus_power) if n_plus_idx[i]]
        for row_idx in row_list:
            row = hidden_weight[row_idx, :]
            edge_list = np.where(row>0)[0]
            model += (pulp.lpSum([V[idx] for idx in edge_list if n_idx[idx]])<=1)

    if n % 2 == 0:
        compare_idx = np.array([1 for i in range(n_power) if n_vt_sum_array[i]%(n+1) == 0 and n_vt_sum_array[i]<=vt_bound])
    else:
        compare_idx = np.array([1 for i in range(n_power) if n_vt_sum_array[i]%(n+1) == (n+1)//2 and n_vt_sum_array[i]<=vt_bound])
    compare = sum(compare_idx)
    

    print(f'n = {n}, Solving LP')
    start = time.time()
    solver = solvers.COIN_CMD('cbc', threads=29)
    model.solve(solver)
    end = time.time()
    opt = value(model.objective)
    print(f'n = {n}, opt = {opt}, compare = {compare} , status: {LpStatus[model.status]}, it took {end-start} seconds')

    for v in model.variables():
        if verbose:
            print(f'{v.name} = {v.varValue}')

    return opt 


if __name__ == "__main__":
    n = 11
    k = 4
    if n % 2 == 0:
        vt_bound = k*(n+1)
    else:
        vt_bound = k*(n+1) + (n+1)//2

    parallel_ip_with_vt_bound(n, vt_bound, inc_plus=False, verbose=False)

