# deletion_graph

import numpy as np

from scipy import sparse, io


def num_to_binstring(idx, n_dim):
    return ("{0:0" + str(n_dim) + "b}").format(idx)



def update_edge(g_edge, idx, n_dim):
    binstring = num_to_binstring(idx, n_dim)
    for i in range(n_dim):
        deleted_string = binstring[:i] + binstring[i+1:]
        for j in range(n_dim):
            for bit in ["0", "1"]:
                inserted_string = deleted_string[:j] + bit + deleted_string[j:]
                out_idx = int(inserted_string, 2)
                if idx != out_idx:
                    g_edge[idx, out_idx] = True 


def write_matrix_market(fname, n_dim):
    n_power = np.power(2, n_dim)
    g_edge = sparse.lil_matrix((n_power, n_power), dtype=bool)

    for idx in range(n_power):
        update_edge(g_edge, idx, n_dim)

    v_num = n_power
    e_num = int(g_edge.count_nonzero()/2)
    
    with open(fname, 'w') as f:
        f.write(f' {v_num} {e_num:}\n')
        for i in range(v_num):
            row_str = ""
            for j in range(v_num):
                if g_edge[i, j]:
                    if len(row_str) == 0:
                        row_str = f"{j+1}"
                    else:
                        row_str += f" {j+1}"
            f.write(row_str)
            if i < v_num-1:
                f.write('\n')

    # writing matrix market file
    # io.mmwrite(fname, g_edge, field='pattern', symmetry='general')


if __name__ == "__main__":
    n_min = 11
    n_max = 15
    n_dim_list = np.arange(n_min, n_max+1)
    for n_dim in n_dim_list:
        fname = f"edge_n{n_dim}_metis.graph"
        print(f"generating {fname}")
        write_matrix_market(fname, n_dim)
