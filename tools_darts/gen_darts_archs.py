import numpy as np
import pickle
import os
import sys
sys.path.append(os.getcwd())
from nas_lib.utils.comm import set_random_seed
from collections import namedtuple
from hashlib import sha256
import argparse

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

NUM_VERTICES = 4
OPS = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]


OPS_WO_NONE = [
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]


def get_gen_hash_key(gentype):
    return sha256(str(gentype).encode('utf-8')).hexdigest()


def gen_arch(ops_list):
    normal = []
    reduction = []
    for i in range(NUM_VERTICES):
        ops = np.random.choice(range(len(ops_list)), NUM_VERTICES)

        # input nodes for conv
        nodes_in_normal = np.random.choice(range(i + 2), 2, replace=False)
        # input nodes for reduce
        nodes_in_reduce = np.random.choice(range(i + 2), 2, replace=False)

        normal.extend([(ops_list[ops[0]], nodes_in_normal[0]), (ops_list[ops[1]], nodes_in_normal[1])])
        reduction.extend([(ops_list[ops[2]], nodes_in_reduce[0]), (ops_list[ops[3]], nodes_in_reduce[1])])
    genotype = Genotype(
        normal=normal,
        normal_concat=[2, 3, 4, 5],
        reduce=reduction,
        reduce_concat=[2, 3, 4, 5]
    )
    return genotype


def gen_arch_wo_key_lists(nums, ops_list, save_path=None):
    archs_list = []
    hash_keys_list = []
    for _ in range(nums):
        if len(archs_list) % 1000 == 0 and len(archs_list) != 0:
            print(f'{len(archs_list)} architectures have generated!')
        genotype = gen_arch(ops_list)
        gen_hash_key = get_gen_hash_key(genotype)

        archs_list.append(genotype)
        hash_keys_list.append(gen_hash_key)
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(archs_list, f)
            pickle.dump(hash_keys_list, f)
    return archs_list


def gen_arch_wo_key_lists(nums, ops_list, save_path=None):
    archs_list = []
    hash_keys_list = []
    for _ in range(nums):
        if len(archs_list) % 1000 == 0 and len(archs_list) != 0:
            print(f'{len(archs_list)} architectures have generated!')
        genotype = gen_arch(ops_list)
        gen_hash_key = get_gen_hash_key(genotype)
        archs_list.append(genotype)
        hash_keys_list.append(gen_hash_key)
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(archs_list, f)
            pickle.dump(hash_keys_list, f)
    return archs_list


def convert_genotype_form(genotype, ops_list):
    normal_cell = genotype.normal
    reduce_cell = genotype.reduce
    normal_cell_new = [(cell[1], ops_list.index(cell[0])) for cell in normal_cell]
    reduce_cell_new = [(cell[1], ops_list.index(cell[0])) for cell in reduce_cell]
    genotype_new = Genotype(
        normal=normal_cell_new,
        reduce=reduce_cell_new,
        normal_concat=genotype.normal_concat,
        reduce_concat=genotype.reduce_concat
    )
    return genotype_new


if __name__ == '__main__':
    # Running this code have to modify the predictor_list, search_space, trails, seq_len, gpu, load_dir, save_dir
    parser = argparse.ArgumentParser(description='Predictor comparison parameters!')
    parser.add_argument('--nums', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=112)
    parser.add_argument('--save_dir', type=str,
                        default='/home/aurora/data_disk_new/train_output_2021/darts_save_path/darts.pkl')

    args = parser.parse_args()
    set_random_seed(args.seed)

    total_archs = gen_arch_wo_key_lists(args.nums, OPS, args.save_dir)
