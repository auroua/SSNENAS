# Copyright (c) Xidian University and Xi'an University of Posts & Telecommunications. All Rights Reserved

import os
import sys
sys.path.append(os.getcwd())
from nas_lib.data.nasbench_201_api import NASBench201API as API
from configs import nas_bench_201_path
from nas_lib.data.nasbench_201_api.genotypes import Structure as CellStructure
from nas_lib.data.nasbench_201_cell import Cell
import numpy as np
from nas_lib.utils.utils_data import find_isolate_node
import pickle
from configs import nas_bench_201_converted_path
import argparse
import copy


NUM_VERTICES = 8


def is_contain_isolate_node(adjacency_matrix):
    idx = 0
    for i in range(len(adjacency_matrix)):
        if np.all(adjacency_matrix[i, :] == 0) and np.all(adjacency_matrix[:, i] == 0):
            if i == 0:
                continue
            idx = i
            break
    return idx


def exchange_nodes_edges(genetype_data):
    global isolate_nums
    ops = ['input']
    data_list = []
    for k in genetype_data:
        data_list.append(k)
    ops.append(data_list[0][0][0])  # 0--->1
    ops.append(data_list[1][0][0])  # 0--->2
    ops.append(data_list[2][0][0])  # 0--->3
    ops.append(data_list[1][1][0])  # 1--->4
    ops.append(data_list[2][1][0])  # 1--->5
    ops.append(data_list[2][2][0])  # 2--->6
    ops.append('output')

    adjacency_matrix = np.zeros((8, 8))
    adjacency_matrix[0, 1] = 1
    adjacency_matrix[0, 2] = 1
    adjacency_matrix[0, 3] = 1
    adjacency_matrix[1, 4] = 1
    adjacency_matrix[1, 5] = 1
    adjacency_matrix[2, 6] = 1
    adjacency_matrix[4, 6] = 1
    adjacency_matrix[3, 7] = 1
    adjacency_matrix[5, 7] = 1
    adjacency_matrix[6, 7] = 1

    del_idxs = [id for id, op in enumerate(ops) if op == 'none']
    ops = [op for op in ops if op != 'none']
    # del_idxs = []
    original_matrix = copy.deepcopy(adjacency_matrix)

    counter = 0
    for id in del_idxs:
        temp_id = id - counter
        adjacency_matrix = np.delete(adjacency_matrix, temp_id, axis=0)
        adjacency_matrix = np.delete(adjacency_matrix, temp_id, axis=1)
        counter += 1

    idx = is_contain_isolate_node(adjacency_matrix)
    while idx > 0:
        del ops[idx]
        adjacency_matrix = np.delete(adjacency_matrix, idx, axis=0)
        adjacency_matrix = np.delete(adjacency_matrix, idx, axis=1)
        idx = is_contain_isolate_node(adjacency_matrix)

    counter = 0
    for i in range(len(adjacency_matrix)):
        if np.all(adjacency_matrix[i, :] == 0) and np.all(adjacency_matrix[:, i] == 0):
            if i == 0:
                continue
            counter += 1

    # if len(del_idxs) > 1:
    # # if len(del_idxs) > 1 and counter > 1:
    #     print(ops)
    #     print(adjacency_matrix)
    #     print(original_matrix)
    #     isolate_nums += 1
    #     print('###############################')
    adjacency_matrix_dummy, ops_dummy = add_dummy_node(adjacency_matrix, ops)
    return adjacency_matrix, ops, adjacency_matrix_dummy, ops_dummy


def add_dummy_node(matrix_in, ops_in):
    # {1, 2, 3, 4, 5, 6, 7}
    matrix = np.zeros((NUM_VERTICES, NUM_VERTICES), dtype=np.int8)
    for i in range(matrix_in.shape[0]):
        idxs = np.where(matrix_in[i] == 1)
        for id in idxs[0]:
            if id == matrix_in.shape[0] - 1:
                matrix[i, NUM_VERTICES-1] = 1
            else:
                matrix[i, id] = 1
    ops = ops_in[:(matrix_in.shape[0] - 1)] + ['isolate'] * (NUM_VERTICES - matrix_in.shape[0]) + ops_in[-1:]
    find_isolate_node(matrix)
    return matrix, ops


def get_arch_acc_info(nas_bench, arch, dataname='cifar10-valid'):
    """

    :param nas_bench:
    :param arch:
    :param dataname: choices ['cifar10-valid', 'cifar10', 'cifar100', 'ImageNet16-120']
    :return:
    """
    arch_index = nas_bench.query_index_by_arch(arch)
    assert arch_index >= 0, 'can not find this arch : {:}'.format(arch)
    info = nas_bench.get_more_info(arch_index, dataname, None, use_12epochs_result=False, is_random=False)
    if dataname == 'cifar10':
        raise NotImplementedError('The dataset cifar10 without having validation accuracy does not support at present!')
    else:
        test_acc, valid_acc = info['test-accuracy'], info['valid-accuracy']
    return valid_acc, test_acc


def generate_all_archs(nas_bench, args):
    total_archs = {}
    total_keys = []
    meta_archs = nas_bench.meta_archs
    for arch in meta_archs:
        val_acc, test_acc = get_arch_acc_info(nas_bench, arch, dataname=args.dataname)
        structure = CellStructure.str2structure(arch)
        am, ops, am_dummy, ops_dummy = exchange_nodes_edges(structure)
        cell_arch = Cell(matrix=am_dummy, ops=ops_dummy, isolate_node_idxs=[])
        path_encoding1 = cell_arch.encode_paths()
        path_encoding2 = cell_arch.encode_cell()
        path_encoding3 = cell_arch.encode_paths_seq_aware(length=96)
        total_archs[arch] = [
            (am_dummy, ops_dummy, []),
            am,
            ops,
            path_encoding1,
            100 - val_acc,
            100 - test_acc,
            arch,
            path_encoding2,
            path_encoding3
        ]
        total_keys.append(arch)
    val_acc = [arch_info[4] for arch_info in total_archs]
    test_acc = [arch_info[5] for arch_info in total_archs]
    print(max(val_acc), min(val_acc), max(test_acc), min(test_acc))
    return total_archs, total_keys


def inti_nasbench_201(args):
    nas_bench = API(nas_bench_201_path)
    total_archs, total_keys = generate_all_archs(nas_bench, args)
    file_name = args.dataname.replace('-', '_')
    save_path = os.path.join(nas_bench_201_converted_path, 'arch_info_'+file_name+'.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(total_archs, f)
        pickle.dump(total_keys, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args for algorithms compare.')
    parser.add_argument('--dataname', type=str, default='cifar10-valid',
                        choices=['cifar10-valid', 'cifar10', 'cifar100', 'ImageNet16-120'],
                        help='Number of trials')

    args = parser.parse_args()
    inti_nasbench_201(args)
