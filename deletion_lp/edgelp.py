# doublehiddenlp.py
import numpy as np

from deletion_utils.bin_num_util import bin_to_num, num_to_bin
from deletion_utils.mis_util import *

from pulp import *


# DUAL
def edge_lp(n, verbose=False, binary=False):
    print(f'n = {n}, Formulating LP')
    n_power = np.power(2, n)
    n_minus_power = np.power(2, n-1)
    n_plus_power = np.power(2, n+1)

    model = LpProblem("Edge LP", LpMinimize)
    if binary:
        cat = 'Binary'
    else:
        cat = 'Continuous'

    E_minus = LpVariable.dicts("E_minus", [i for i in range(n_minus_power)], lowBound=0, cat=cat)
    E_plus = LpVariable.dicts("E_plus", [i for i in range(n_plus_power)], lowBound=0, cat=cat)

    model += (pulp.lpSum(E_minus) + pulp.lpSum(E_plus))
    
    minus_hidden_weight = get_hidden_weight_matrix(n)
    plus_hidden_weight = get_hidden_weight_matrix(n+1)
    for idx in range(0, n_power):
        minus_edge = minus_hidden_weight[idx, :]
        minus_edge_list = np.where(minus_edge>0)[0]

        plus_edge = plus_hidden_weight[:, idx]
        plus_edge_list = np.where(plus_edge>0)[0]

        bin_str = num_to_bin(idx, n)
        if compute_vt_sum(bin_str) == 0:
            model += (pulp.lpSum([E_minus[edge] for edge in minus_edge_list] + [E_plus[edge] for edge in plus_edge_list]) == 1)
        else:
            model += (pulp.lpSum([E_minus[edge] for edge in minus_edge_list] + [E_plus[edge] for edge in plus_edge_list]) == 1)

    print(f'n = {n}, Solving LP')
    model.solve()
    opt = value(model.objective)
    print(f'n={n}, opt={opt}, binary={binary}, status: {LpStatus[model.status]}')


if __name__ == "__main__":
    for binary in [True]:
    # for binary in [False]:
        for n in range(6, 12):
            edge_lp(n, verbose=False, binary=binary)
