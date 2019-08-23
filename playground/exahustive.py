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
    for i in range(n_power):
        # append edge array (edge from V_n to V_{n-1})
        edge_arr.append(set(np.nonzero(edges_from_vertex(i, n))[0]))

        # build vt database
        bin_str = num_to_bin(i, n)
        vt_sum = compute_vt_sum(bin_str, raw=True)
        if vt_sum not in vt_db:
            vt_db[vt_sum] = [i]
        else:
            vt_db[vt_sum].append(i)

    vt_db = dict(sorted(vt_db.items()))
    return vt_db, edge_arr

if __name__ == "__main__":
    n = 3
    vt_db, edge_arr = build_db(n+1)
    print(vt_db)
    print(edge_arr)

    m = int((n+2)*(n+1)/2)
    v_set = [[0]]
    for idx, s_idx in enumerate(range(n+2, m+1, n+2), 1):
        prev_idx = idx-1
        print(vt_db[s_idx])

    vt0 = get_vt0(n)
    print(f"VT Code has {len(vt0)} elements")

