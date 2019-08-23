# greedy.py

import numpy as np

from mis_util import *
from bin_num_util import *


def build_db(n):
    n_power = np.power(2, n)

    vt_db = {}
    weight_matrix = get_weight_matrix(n)
    np.fill_diagonal(weight_matrix, 0)
    edge_arr = []
    # for i in range(n_power):
    #     edges = edges_from_vertex(i, n)
    #     edge_arr.append(np.nonzero(edges)[0])
    for i in range(n_power):
        # append edge array
        edge_arr.append(set(np.nonzero(weight_matrix[i])[0]))
        # build vt database
        bin_str = num_to_bin(i, n)
        vt_sum = compute_vt_sum(bin_str, raw=True)
        if vt_sum not in vt_db:
            vt_db[vt_sum] = [i]
        else:
            vt_db[vt_sum].append(i)
    # edge_arr = np.nonzero(weight_matrix)
    vt_db = dict(sorted(vt_db.items()))
    return vt_db, edge_arr

if __name__ == "__main__":
    # n = 15
    # vt_db, edge_arr = build_db(n)
    # print("got db")

    # m = int(n*(n+1)/2)
    # 
    # indep_set = [] 
    # edge_set = set([]) 
    # for vt_sum in range(0, m+1, 8):
    #     print(f"vt_sum={vt_sum}")
    #     for vertex in vt_db[vt_sum]:
    #         if vertex not in edge_set:
    #             indep_set.append(vertex)
    #             edge_set = edge_set.union(edge_arr[vertex])
    # print(len(indep_set))

    for n in range(4, 16):
        vt0 = get_vt0(n)
        print(f"n={n}: VT Code has {len(vt0)} elements")

