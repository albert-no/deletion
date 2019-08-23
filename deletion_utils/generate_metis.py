# generate_metis.py

import numpy as np

from deletion_utils.mis_util import *
from deletion_utils.bin_num_util import *

if __name__ == "__main__":
    n = 11
    vt_list = list(range(30, 31))
    for vt_up in vt_list:
        hidden_weight = get_hidden_weight_matrix(n, vt_up=vt_up)
        weight = hidden_weight.dot(hidden_weight.T)
        np.fill_diagonal(weight, 0)

        fname = f'n{n}_vt{vt_up}_metis.graph'

        v_num = len(weight)
        e_num = np.count_nonzero(weight)//2
        with open(fname, 'w') as f:
            f.write(f' {v_num} {e_num}\n')
            for i in range(v_num):
                row_str = ""
                for j in range(v_num):
                    if weight[i, j] > 0:
                        if len(row_str) == 0:
                            row_str = f"{j+1}"
                        else:
                            row_str += f" {j+1}"
                f.write(row_str)
                if i < v_num-1:
                    f.write('\n')

