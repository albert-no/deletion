# greedy_hiddenlp.py
import numpy as np
import pandas as pd

from deletion_utils.bin_num_util import bin_to_num, num_to_bin
from deletion_utils.mis_util import get_hidden_weight_matrix, edges_from_vertex

from pulp import *


def check_node(n, name, mis, hidden_weight):
    current_edge = edges_from_vertex(name, n)
    for mis_node in mis:
        mis_node_edge = hidden_weight[mis_node, :]
        if mis_node_edge.T.dot(current_edge) > 0:
            return False
    return True

def greedy_hidden_lp(n, verbose=False):
    print(f'n = {n}, Formulating LP')
    n_power = np.power(2, n)
    n_minus_power = np.power(2, n-1)

    hidden_weight = get_hidden_weight_matrix(n, normalize=False)

    lp_sol_fname = f'./lpsols/converted_lp_sol_n{n}.txt'
    df_lp = pd.read_csv(lp_sol_fname)

    df_lp_sort = df_lp.sort_values(['val'], ascending=False)

    mis = []
    for _, row in df_lp_sort.iterrows():
        name = int(row['name'])
        if check_node(n, name, mis, hidden_weight):
            mis.append(int(name))
    print(mis)
    print(len(mis))



if __name__ == "__main__":
    # for n in range(4, 6):
    n = 6
    greedy_hidden_lp(n, verbose=False)
