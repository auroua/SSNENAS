# Copyright (c) Xidian University and Xi'an University of Posts & Telecommunications. All Rights Reserved

import numpy as np
import torch
import math


NUM_VERTICES_101 = 7
NASBENCH_101_OPS = {
    'input': 0,
    'conv3x3-bn-relu': 1,
    'conv1x1-bn-relu': 2,
    'maxpool3x3': 3,
    'output': 4,
    'isolate': 5
}

NASBENCH_101_OPS_LIST = ['input', 'conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3', 'output', 'isolate']

NUM_VERTICES_201 = 8
NASBENCH_201_OPS = ['input', 'none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3',
                    'avg_pool_3x3', 'isolate', 'output']

NUM_VERTICES_DARTS = 30
DARTS_OPS = ['input', 'none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5',
             'dil_conv_3x3', 'dil_conv_5x5', 'concat', 'output']


DARTS_OPS_DICT = {'input': 0,
                  'none': 1,
                  'max_pool_3x3': 2,
                  'avg_pool_3x3': 3,
                  'skip_connect': 4,
                  'sep_conv_3x3': 5,
                  'sep_conv_5x5': 6,
                  'dil_conv_3x3': 7,
                  'dil_conv_5x5': 8,
                  'concat': 9,
                  'output': 10
                  }
NUM_VERTICES_DARTS_HALF = 15


def find_isolate_node(matrix):
    node_list = []
    for i in range(len(matrix)):
        if np.all(matrix[i, :] == 0) and np.all(matrix[:, i] == 0):
            if i == 0:
                continue
            matrix[0, i] = 1
            node_list.append(i)
    return node_list


def nasbench2graph_101(data, is_idx=False, reverse=False):
    matrix, ops = data[0], data[1]
    if reverse:
        matrix = matrix.T
    node_feature = torch.zeros(NUM_VERTICES_101, 6)
    if isinstance(matrix, torch.Tensor):
        edges = int(torch.sum(matrix).item())
    else:
        edges = int(np.sum(matrix))
    edge_idx = torch.zeros(2, edges)
    counter = 0
    for i in range(NUM_VERTICES_101):
        if is_idx:
            idx = int(ops[i].item())
        else:
            idx = NASBENCH_101_OPS[ops[i]]
        node_feature[i, idx] = 1
        for j in range(NUM_VERTICES_101):
            if matrix[i, j] == 1:
                edge_idx[0, counter] = i
                edge_idx[1, counter] = j
                counter += 1
    return edge_idx, node_feature


def nasbench2graph_201(data, is_idx=False, reverse=False):
    matrix, ops = data[0], data[1]
    if reverse:
        matrix = matrix.T
    node_feature = torch.zeros(NUM_VERTICES_201, 8)
    if isinstance(matrix, torch.Tensor):
        edges = int(torch.sum(matrix).item())
    else:
        edges = int(np.sum(matrix))
    edge_idx = torch.zeros(2, edges)
    counter = 0
    for i in range(NUM_VERTICES_201):
        if is_idx:
            idx = int(ops[i].item())
        else:
            idx = NASBENCH_201_OPS.index(ops[i])
        node_feature[i, idx] = 1
        for j in range(NUM_VERTICES_201):
            if matrix[i, j] == 1:
                edge_idx[0, counter] = i
                edge_idx[1, counter] = j
                counter += 1
    return edge_idx, node_feature


def nasbench2graph_darts(data, reverse=False):
    matrix, ops = data[0], data[1]
    if reverse:
        matrix = matrix.T
    node_feature = torch.zeros(NUM_VERTICES_DARTS_HALF, 11)
    edges = int(np.sum(matrix))
    edge_idx = torch.zeros(2, edges)
    counter = 0
    for i in range(NUM_VERTICES_DARTS_HALF):
        idx = DARTS_OPS_DICT[ops[i]]
        node_feature[i, idx] = 1
        for j in range(NUM_VERTICES_DARTS_HALF):
            if matrix[i, j] == 1:
                edge_idx[0, counter] = i
                edge_idx[1, counter] = j
                counter += 1
    return edge_idx, node_feature


def nas2graph(nas_benchmark, data, reverse=False):
    if nas_benchmark == 'nasbench_101':
        return nasbench2graph_101(data, reverse=reverse)
    elif nas_benchmark == 'nasbench_201':
        return nasbench2graph_201(data, reverse=reverse)
    elif nas_benchmark == 'darts':
        return nasbench2graph_darts(data, reverse=reverse)
    else:
        raise NotImplementedError(f'The nas benchmark type {nas_benchmark} have not implemented yet!')


def get_node_num(nas_benchmark):
    if nas_benchmark == 'nasbench_101':
        return NUM_VERTICES_101
    elif nas_benchmark == 'nasbench_201':
        return NUM_VERTICES_201
    elif nas_benchmark == 'darts':
        return NUM_VERTICES_DARTS
    else:
        raise NotImplementedError(f'The nas benchmark type {nas_benchmark} have not implemented yet!')


def get_node_type_num(nas_benchmark):
    if nas_benchmark == 'nasbench_101':
        return len(NASBENCH_101_OPS)
    elif nas_benchmark == 'nasbench_201':
        return len(NASBENCH_201_OPS)
    elif nas_benchmark == 'darts':
        return len(DARTS_OPS)
    else:
        raise NotImplementedError(f'The nas benchmark type {nas_benchmark} have not implemented yet!')


def get_ops_list(nas_benchmark):
    if nas_benchmark == 'nasbench_101':
        return NASBENCH_101_OPS_LIST
    elif nas_benchmark == 'nasbench_201':
        return NASBENCH_201_OPS
    elif nas_benchmark == 'darts':
        return DARTS_OPS
    else:
        raise NotImplementedError(f'The nas benchmark type {nas_benchmark} have not implemented yet!')


def get_input_dim(search_space):
    if search_space == 'nasbench_101':
        return 6
    elif search_space == 'nasbench_201':
        return 8
    elif search_space == 'darts':
        return 11
    else:
        raise NotImplementedError(f'The search space type {search_space} does not support!')


def get_seq_len(search_space):
    if search_space == 'nasbench_101':
        return 120
    elif search_space == 'nasbench_201':
        return 96
    elif search_space == 'darts':
        return 1224
    else:
        raise NotImplementedError(f'The search space {search_space} does not support!')


def edit_distance_normalization(path_encoding_1, path_endocing_2, num_nodes):
    distance = np.sum(np.array(path_encoding_1) != np.array(path_endocing_2)) * 1.0
    distance = math.exp(-1.*(distance/num_nodes))
    return distance


def edit_distance(path_encoding_1, path_endocing_2):
    distance = np.sum(np.array(path_encoding_1) != np.array(path_endocing_2)) * 1.0
    return distance


def generate_min_vals(p1, p2, return_matrix=False):
    p1 = p1.view(1, p1.size(0), p1.size(1))
    p2 = p2.view(p2.size(0), 1, p2.size(1))
    dist = torch.sum(torch.abs(p1 - p2), dim=-1).int().T
    eigen_index = [i for i in range(dist.size(0))]
    dist[eigen_index, eigen_index] = 100
    min_vals, min_indices = torch.min(dist, dim=1)
    if return_matrix:
        return dist
    else:
        return min_vals, min_indices


def analysis_matrix(dist_matrix):
    dist_matrix_np = dist_matrix.numpy()
    for i in range(dist_matrix_np.shape[0]):
        min_val = np.min(dist_matrix_np[i, :])
        print(min_val, np.sum(dist_matrix_np[i, :] == min_val))


def nasbench2graph_reverse(data, reverse=False):
    OPS = {
        'input': 0,
        'conv3x3-bn-relu': 1,
        'conv1x1-bn-relu': 2,
        'maxpool3x3': 3,
        'output': 4,
        'isolate': 5
    }

    NUM_VERTICES = 7

    matrix, ops = data[0], data[1]
    if reverse:
        matrix = matrix.T
    node_feature = torch.zeros(NUM_VERTICES, 6)
    edges = int(np.sum(matrix))
    edge_idx = torch.zeros(2, edges)
    counter = 0
    for i in range(NUM_VERTICES):
        idx = OPS[ops[i]]
        node_feature[i, idx] = 1
        for j in range(NUM_VERTICES):
            if matrix[i, j] == 1:
                edge_idx[0, counter] = i
                edge_idx[1, counter] = j
                counter += 1
    return edge_idx, node_feature


def gen_batch_idx(idx_list, batch_size):
    ds_len = len(idx_list)
    idx_batch_list = []

    for i in range(0, math.ceil(ds_len/batch_size)):
        if (i+1)*batch_size > ds_len:
            idx_batch_list.append(idx_list[i*batch_size:])
        else:
            idx_batch_list.append(idx_list[i*batch_size: (i+1)*batch_size])
    return idx_batch_list