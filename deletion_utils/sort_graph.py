# graph_image.py

import numpy as np
from deletion_utils.mis_util import *


def get_s_vector(n):
    prev_s_matrix = [[0], [1]]
    for i in range(2, n+1):
        m = int(i*(i+1)/2)
        s_matrix = []
        i_power = int(np.power(2, i-1))
        for j in range(m+1):
            if j < i:
                s_matrix.append(prev_s_matrix[j])
            elif j > i*(i-1)/2:
                s_row = [x+i_power for x in prev_s_matrix[j-i]]
                s_matrix.append(s_row)
            else:
                s_row0 = prev_s_matrix[j]
                s_row1 = [x+i_power for x in prev_s_matrix[j-i]]
                s_matrix.append(s_row1+s_row0)
        if i < n:
            prev_s_matrix = s_matrix.copy()
    prev_s_vector = sum(prev_s_matrix, [])
    s_vector = sum(s_matrix, [])
    return prev_s_vector, s_vector


def sort_graph(n):
    hidden_weight_matrix = get_hidden_weight_matrix(n)
    prev_s_vector, s_vector = get_s_vector(n)
    row_swap = hidden_weight_matrix[s_vector, :]
    col_swap = row_swap[:, prev_s_vector]
    return col_swap
