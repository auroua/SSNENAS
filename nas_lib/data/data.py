# Copyright (c) Xidian University and Xi'an University of Posts & Telecommunications. All Rights Reserved

import random
from .nasbench_101_cell import Cell as Cell_101
from .nasbench_201_cell import Cell as Cell_201
from gnn_lib.data import Data
from nas_lib.utils.utils_data import nas2graph


def build_datasets(args):
    if args.search_space == "nasbench_101":
        from nas_lib.data.nasbench_101 import NASBench101
        return NASBench101(args.search_space)
    elif args.search_space == 'nasbench_201':
        from nas_lib.data.nasbench_201 import NASBench201
        return NASBench201(args)
    else:
        raise ValueError("This architecture datasets does not support!")


def dataset_split(args, nas_dataset, budget=None):
    total_keys = nas_dataset.total_keys
    total_archs = nas_dataset.total_archs
    if budget:
        train_keys = random.sample(total_keys, budget)
    else:
        train_keys = random.sample(total_keys, args.search_budget)
    test_keys = [key for key in total_keys if key not in train_keys]
    train_data = []
    test_data = []

    flag = args.search_space == 'nasbench_101'
    for k in train_keys:
        arch = total_archs[k]
        if args.search_space == 'nasbench_101':
            cell_inst = Cell_101(matrix=arch['matrix'], ops=arch['ops'])
        elif args.search_space == 'nasbench_201':
            cell_inst = Cell_201(matrix=arch[0][0], ops=arch[0][1])
        else:
            raise NotImplementedError()
        train_data.append(
            {
                'matrix': arch['matrix'] if flag else arch[0][0],
                'ops': arch['ops'] if flag else arch[0][1],
                'pe_adj_enc_vec': cell_inst.get_encoding('adj_enc_vec', args.seq_len),
                'pe_path_enc_vec': cell_inst.get_encoding('path_enc_vec', args.seq_len),
                'pe_path_enc_aware_vec': cell_inst.get_encoding('path_enc_aware_vec', args.seq_len),
                'val_acc': arch['val'] if flag else (100-arch[4]) * 0.01,
                'test_acc': arch['test'] if flag else (100-arch[5]) * 0.01
            }
        )

    for k in test_keys:
        arch = total_archs[k]
        if args.search_space == 'nasbench_101':
            cell_inst = Cell_101(matrix=arch['matrix'], ops=arch['ops'])
        elif args.search_space == 'nasbench_201':
            cell_inst = Cell_201(matrix=arch[0][0], ops=arch[0][1])
        else:
            raise NotImplementedError()
        test_data.append(
            {
                'matrix': arch['matrix'] if flag else arch[0][0],
                'ops': arch['ops'] if flag else arch[0][1],
                'pe_adj_enc_vec': cell_inst.get_encoding('adj_enc_vec', args.seq_len),
                'pe_path_enc_vec': cell_inst.get_encoding('path_enc_vec', args.seq_len),
                'pe_path_enc_aware_vec': cell_inst.get_encoding('path_enc_aware_vec', args.seq_len),
                'val_acc': arch['val'] if flag else (100-arch[4]) * 0.01,
                'test_acc': arch['test'] if flag else (100-arch[5]) * 0.01
            }
        )
    return train_data, test_data


def dataset_all(args, nas_dataset):
    total_keys = nas_dataset.total_keys
    total_archs = nas_dataset.total_archs
    all_archs = []

    flag = args.search_space == 'nasbench_101'
    for k in total_keys:
        arch = total_archs[k]
        if args.search_space == 'nasbench_101':
            cell_inst = Cell_101(matrix=arch['matrix'], ops=arch['ops'])
            edge_index, node_f = nas2graph(args.search_space, (arch['matrix'], arch['ops']))
        elif args.search_space == 'nasbench_201':
            cell_inst = Cell_201(matrix=arch[0][0], ops=arch[0][1])
            edge_index, node_f = nas2graph(args.search_space, (arch[0][0], arch[0][1]))
        else:
            raise NotImplementedError()

        all_archs.append(
            {
                'matrix': arch['matrix'] if flag else arch[0][0],
                'ops': arch['ops'] if flag else arch[0][1],
                'pe_adj_enc_vec': cell_inst.get_encoding('adj_enc_vec', args.seq_len),
                'pe_path_enc_vec': cell_inst.get_encoding('path_enc_vec', args.seq_len),
                'pe_path_enc_aware_vec': cell_inst.get_encoding('path_enc_aware_vec', args.seq_len),
                'val_acc': arch['val'] if flag else (100-arch[4]) * 0.01,
                'test_acc': arch['test'] if flag else (100-arch[5]) * 0.01,
                'g_data': Data(edge_index=edge_index.long(), x=node_f.float())
            }
        )
    return all_archs


def split_data_from_all_data(all_data, idxs, train_data, budget, last_budget):
    train_data_new = []
    counter = 0
    while len(train_data_new) < (budget - last_budget):
        if idxs[last_budget+counter] < len(all_data):
            train_data_new.append(all_data.pop(idxs[last_budget+counter]))
            counter += 1
        else:
            counter += 1
            continue
    train_data.extend(train_data_new)
    return train_data, all_data


def dataset_split_idx(all_data, budget=None):
    idxs = list(range(len(all_data)))
    random.shuffle(idxs)
    train_data = [all_data[k] for k in idxs[:budget]]
    test_data = [all_data[kt] for kt in idxs[budget:]]
    return train_data, test_data