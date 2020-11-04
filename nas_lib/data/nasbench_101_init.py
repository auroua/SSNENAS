import pickle
import argparse
import os
import sys
sys.path.append(os.getcwd())
from configs import nas_bench_101_base_path
from nas_lib.data import data


def generate_nasbench_101_bench_keys_vals(nasbench_data):
    save_path = os.path.join(nas_bench_101_base_path, 'nasbench_archs.pkl')
    keys = nasbench_data.total_keys
    archs = nasbench_data.total_archs
    with open(save_path, 'wb') as fw:
        pickle.dump(keys, fw)
        pickle.dump(archs, fw)


def generate_nasbench_101_all_datas(nasbench_data, args):
    all_data = data.dataset_all(args, nasbench_data)
    save_path = os.path.join(nas_bench_101_base_path, 'all_data_new.pkl')
    with open(save_path, 'wb') as fb:
        pickle.dump(all_data, fb)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args for NASBench_101 init!')
    #  unsupervised_ged: SS_RL
    parser.add_argument('--search_space', type=str, default='nasbench_101',
                        choices=['nasbench_101'],
                        help='The search space.')
    args = parser.parse_args()
    args.seq_len = 120
    nasbench_datasets = data.build_datasets(args)
    generate_nasbench_101_bench_keys_vals(nasbench_datasets)
    generate_nasbench_101_all_datas(nasbench_datasets, args)

