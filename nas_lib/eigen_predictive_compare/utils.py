import math
from collections import defaultdict
import torch


def gen_batch_idx(idx_list, batch_size):
    ds_len = len(idx_list)
    idx_batch_list = []

    for i in range(0, math.ceil(ds_len/batch_size)):
        if (i+1)*batch_size > ds_len:
            idx_batch_list.append(idx_list[i*batch_size:])
        else:
            idx_batch_list.append(idx_list[i*batch_size: (i+1)*batch_size])
    return idx_batch_list
