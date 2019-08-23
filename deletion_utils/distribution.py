# distribution.py

import numpy as np
from deletion_utils.bin_num_util import num_to_bin, v_num_to_bin
from deletion_utils.mis_util import compute_vt_sum ,v_compute_vt_sum, get_vt0

sol_fname_template = '../sols/sols-{}.txt'
np.set_printoptions(linewidth=100)

def get_count_arr(sol, num):
    sol_bin = v_num_to_bin(sol, num)
    vt_sum = v_compute_vt_sum(sol_bin, raw=True)
    unique, counts = np.unique(vt_sum, return_counts=True)
    count_dict = dict(zip(unique, counts))
    # print(vt_count)
    count_arr = np.zeros(int(num*(num+1)/2)+1)
    count_arr[unique] = counts
    return count_arr


for num in [11]:
    n_power = np.power(2, num)
    # sol_fname = sol_fname_template.format(n_power)
    # sols= np.genfromtxt(sol_fname, skip_header=1, dtype=int)
    # for sol in sols:
    #     count_arr = get_count_arr(sol, num)
    #     # print(count_arr)
    #     print([sum(count_arr[:(num+1)*i+1]) for i in range((num+2)//2)])
    # vt0 = get_vt0(num)
    # count_arr = get_count_arr(vt0, num)
    # print(count_arr)
    # print([sum(count_arr[:(num+1)*i+1]) for i in range((num+2)//2)])
    total = np.array(range(n_power))
    count_arr = get_count_arr(total, num)
    m = num*(num+1)//2
    # for j in range(m+1):
    for j in range(31):
        print(sum(count_arr[j::-(num+1)]))
    print(count_arr)


# n = 12
# n_power = np.power(2, n)
# total = np.array(range(n_power))
# count_arr = get_count_arr(total, n)
# print(count_arr)
# m = n*(n+1)/2
# a = 1
# while a <= m:
#     print(f"VT_{a}: {count_arr[a]}")
#     a += (n+1)
# 
# 
#     
