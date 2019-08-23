# graph_image.py

import numpy as np
from PIL import Image
from matplotlib import pylab as pt
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
    row_ind = []
    for idx, s_element in enumerate(s_matrix):
        if idx % 2 == 0:
            row_ind += [1] * len(s_element)
        else:
            row_ind += [0] * len(s_element)
    col_ind = []
    for idx, prev_s_element in enumerate(prev_s_matrix):
        if idx % 2 == 0:
            col_ind += [1] * len(prev_s_element)
        else:
            col_ind += [0] * len(prev_s_element)

    return prev_s_vector, s_vector, row_ind, col_ind


if __name__ == "__main__":
    n_max = 9
    for n in range(3, n_max+1):
        hidden_weight_matrix = get_hidden_weight_matrix(n)
        # print(hidden_weight_matrix)
        prev_s_vector, s_vector, row_ind, col_ind = get_s_vector(n)
        row_swap = hidden_weight_matrix[s_vector, :]
        col_swap = row_swap[:, prev_s_vector]

        row_idx = [i for i, x in enumerate(row_ind) if x == 1]
        col_idx = [i for i, x in enumerate(col_ind) if x == 1]
        col_swap[row_ind, :] = (col_swap[row_ind, :] + 1)//2
        col_swap[:, col_ind] = (col_swap[:, col_ind] + 1)//2
        # im = Image.fromarray(col_swap)
        # im.save(f'n{n}.tif')
        fig, ax = pt.subplots()
        ax.axis('off')
        for idx, row in enumerate(row_ind):
            if idx > 0 and row_ind[idx-1] != row:
                ax.axhline(y=idx-0.5, color='red', linewidth=0.5)
        for idx, col in enumerate(col_ind):
            if idx > 0 and col_ind[idx-1] != col:
                ax.axvline(x=idx-0.5, color='red', linewidth=0.5)
        # pt.imshow(col_swap)

        # pt.show()
        ax.imshow(col_swap)
        fig.savefig(f'n{n}.pdf')#, col_swap)
