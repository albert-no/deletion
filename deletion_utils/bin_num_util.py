# bin_num_util.py
import numpy as np


def bin_to_num(bin_str):
    return int(bin_str, 2)


def num_to_bin(x, n):
    bin_template = '{:0'+str(n)+'b}'
    bin_str = bin_template.format(x)
    return bin_str


v_bin_to_num = np.vectorize(bin_to_num)
v_num_to_bin = np.vectorize(num_to_bin)


def get_binary_array(n):
    n_power = np.power(2, n)
    n_list = np.arange(n_power)
    n_array = v_num_to_bin(n_list, n)
    return n_array
