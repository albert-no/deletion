# parallel_looplp.py
import numpy as np
import random
import time

from deletion_utils.bin_num_util import bin_to_num, num_to_bin
from deletion_utils.mis_util import *
from deletion_utils.clique import *
from deletion_lp.node_from_parallel import NODES

from pulp import *


def loop_lp(n, verbose=False, v_bin=[0], v_zeros=[1], v_ones=[0]):
    print(f'n = {n}, Formulating LP')
    n_power = np.power(2, n)
    n_minus_power = np.power(2, n-1)
    n_plus_power = np.power(2, n+1)

    model = LpProblem("Loop LP", LpMaximize)

    v_else = [i for i in range(n_power) if i not in v_bin]

    Vbin = [i for i in v_bin]
    Velse = [i for i in v_else]

    VBIN = LpVariable.dicts("VBIN", [i for i in v_bin], lowBound=0, cat='Binary')
    VELSE = LpVariable.dicts("VELSE", [i for i in v_else], lowBound=0, upBound=1)

    model += (pulp.lpSum(VBIN)+pulp.lpSum(VELSE))
    
    for v in v_zeros:
        if v in v_bin:
            model += (VBIN[v] == 0)
        else:
            model += (VELSE[v] == 0)
    for v in v_ones:
        if v in v_bin:
            model += (VBIN[v] == 1)
        else:
            model += (VELSE[v] == 1)

    hidden_weight = get_hidden_weight_matrix(n+1)
    for row_idx in range(0, n_plus_power):
        row = hidden_weight[row_idx, :]
        edge_list = np.where(row>0)[0]
        v_bin_list = [edge for edge in edge_list if edge in v_bin]
        v_else_list = [edge for edge in edge_list if edge in v_else]
        model += ((pulp.lpSum([VBIN[idx] for idx in v_bin_list])+pulp.lpSum([VELSE[idx] for idx in v_else_list])) <= 1)

    hidden_weight = get_hidden_weight_matrix(n)
    for col_idx in range(0, n_minus_power):
        col = hidden_weight[:, col_idx]
        edge_list = np.where(col>0)[0]
        v_bin_list = [edge for edge in edge_list if edge in v_bin]
        v_else_list = [edge for edge in edge_list if edge in v_else]
        model += ((pulp.lpSum([VBIN[idx] for idx in v_bin_list])+pulp.lpSum([VELSE[idx] for idx in v_else_list])) <= 1)

    print(f'n = {n}, Solving LP')
    start = time.time()
    solver = solvers.COIN_CMD('cbc', threads=28)
    model.solve(solver)
    end = time.time()
    opt = value(model.objective)
    print(f'n = {n}, opt = {opt}, status: {LpStatus[model.status]}, it took {end-start} seconds')

    zeros = v_zeros
    ones = v_ones
    binary = v_bin
    temp_rule = 0
    temp_idx = -1
    for v in model.variables():
        idx = v.name.find('_')
        idx_int = int(v.name[idx+1:])
        bin_str = num_to_bin(idx_int, n)
        vt_sum = compute_vt_sum(bin_str)
        if v.varValue == 1:
            if verbose:
                print(f'{v.name} = {v.varValue}, VT_sum = {vt_sum}')
            if idx_int not in binary:
                binary.append(idx_int)
        elif v.varValue > 0:
            rule = v.varValue * sum(hidden_weight[idx_int, :])
            if rule > temp_rule:
                temp_idx = idx_int
                temp_max = v.varValue
                temp_rule = rule
            if verbose:
                print(f'{v.name} = {v.varValue}, VT_sum = {vt_sum}')

    if temp_idx >= 0:
        print(f'v_idx = {temp_idx}, '
              f'v_val = {temp_max}, '
              f'hidden_weight = {sum(hidden_weight[temp_idx, :])}')
        binary.append(temp_idx)
    return binary, opt 


if __name__ == "__main__":
    n = 12
    zeros = ([np.power(2, i) for i in range(n)]
            + [np.power(2, n)-1-np.power(2, i) for i in range(n)])
    # binary = NODES
    binary = []
    cnt = len(binary)
    ones = [0, np.power(2, n)-1]
    while True:
        cnt += 1
        print(' ')
        print(f'{cnt}-th loop')
        binary, current_opt = loop_lp(
                n, verbose=False, v_bin=binary, v_zeros=zeros, v_ones=ones)
