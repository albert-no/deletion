# findclique.py
import numpy as np
import itertools

from deletion_utils.bin_num_util import bin_to_num, num_to_bin
from deletion_utils.mis_util import *

from pulp import *



def findclique(n):
    n_power = np.power(2, n)
    n_minus_power = np.power(2, n-1)
    n_plus_power = np.power(2, n+1)

    minus_hidden_weight = get_hidden_weight_matrix(n)
    plus_hidden_weight = get_hidden_weight_matrix(n+1)

    weight_matrix = get_weight_matrix(n)

    cliques = []
    for i, j, k in itertools.product(range(0, n_power), range(0, n_power), range(0, n_power)):
        if i>=j or j>=k:
            continue
        if weight_matrix[i, j] > 0 and weight_matrix[j, k] > 0 and weight_matrix[k, i] > 0:
            minus_clique = sum(minus_hidden_weight[i, :]* minus_hidden_weight[j, :]* minus_hidden_weight[k, :])
            plus_clique = sum(plus_hidden_weight[:, i]*plus_hidden_weight[:, j]*plus_hidden_weight[:, k])
            if minus_clique == 0 and plus_clique == 0:
                cliques.append([i, j, k])
    return cliques
