# mis_util.py
import numpy as np
from deletion_utils.bin_num_util import (
    bin_to_num, num_to_bin, v_bin_to_num, v_num_to_bin, get_binary_array)


def edges_from_vertex(x, n):
    bin_str = num_to_bin(x, n)
    edge_arr = np.zeros(np.power(2, n-1))
    deleted_bin_arr = [bin_str[:i]+bin_str[i+1:] for i in range(n)]
    edge_arr[v_bin_to_num(deleted_bin_arr)] = 1
    return edge_arr


def get_hidden_weight_matrix(n, normalize=False, transpose=False):
    n_power = np.power(2, n)
    hidden_weight_matrix = np.array([edges_from_vertex(x, n) for x in range(n_power)])
    if normalize:
        row_sum = np.sqrt(np.sum(hidden_weight_matrix, 1))
        hidden_weight_matrix = hidden_weight_matrix / row_sum[:, np.newaxis]

    if transpose:
        hidden_weight_matrix = hidden_weight_matrix.T
    return hidden_weight_matrix


def get_weight_matrix(n, normalize=False):
    hidden_weight_matrix = get_hidden_weight_matrix(n, normalize)
    weight_matrix = hidden_weight_matrix.dot(hidden_weight_matrix.T)
    return weight_matrix


def compute_vt_sum(bin_str, reverse=False, raw=False):
    n = len(bin_str)
    weight = np.arange(1, n+1)
    if reverse:
        weight = weight[::-1]
    vt_sum = sum([weight[i]*int(bin_str[i]) for i in range(n)])
    if raw:
        return vt_sum
    else:
        return np.mod(vt_sum,(n+1))

v_compute_vt_sum = np.vectorize(compute_vt_sum)


# vt0 = vt0(reverse) always
def get_vt0(n, reverse=False, numeric=True):
    n_array = get_binary_array(n)
    vt0_idx = [compute_vt_sum(bin_str, reverse)==0 for bin_str in n_array]
    vt0 = n_array[vt0_idx]
    if numeric:
        vt0 = v_bin_to_num(vt0)
    return vt0


def write_weight_result(f, name, mis, node_weight):
    sel = node_weight[mis]
    f.write(f'{name}: {sum(sel)}\n')
    f.write(str(mis)+'\n')
    f.write(str(sel)+'\n\n')
