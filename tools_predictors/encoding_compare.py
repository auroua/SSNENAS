import argparse
import os
import sys
sys.path.append(os.getcwd())
from configs import nas_bench_101_all_data
import pickle
from nas_lib.data import data


if __name__ == '__main__':
    # Running this code have to modify the predictor_list, search_space, trails, seq_len, gpu, load_dir, save_dir
    parser = argparse.ArgumentParser(description='Predictor comparison parameters!')
    parser.add_argument('--search_space', type=str, default='nasbench_101',
                        choices=['nasbench_101', 'nasbench_201', 'darts'],
                        help='The search space.')
    parser.add_argument('--dataname', type=str, default='cifar10-valid',
                        choices=['cifar10-valid', 'cifar10', 'cifar100', 'ImageNet16-120'],
                        help='The evaluation of dataset of NASBench-201.')
    args = parser.parse_args()

    if args.search_space == 'nasbench_101':
        args.seq_len = 120
        with open(nas_bench_101_all_data, 'rb') as fpkl:
            all_data = pickle.load(fpkl)
    elif args.search_space == 'nasbench_201':
        args.seq_len = 96    # 5461
        nasbench_datas = data.build_datasets(args)
        all_data = data.dataset_all(args, nasbench_datas)
    else:
        raise NotImplementedError('This search space does not support at present!')

    path_based_encoding = [tuple(map(int, d['pe_path_enc_vec'].tolist())) for d in all_data]
    path_based_position_aware_encoding = [tuple(map(int, d['pe_path_enc_aware_vec'].tolist())) for d in all_data]
    print('path_based encoding', len(path_based_encoding), len(set(path_based_encoding)))
    print('position_aware_path_based_encoding', len(path_based_position_aware_encoding), len(set(path_based_position_aware_encoding)))