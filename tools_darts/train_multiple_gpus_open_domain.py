# Copyright (c) XiDian University and Xi'an University of Posts&Telecommunication. All Rights Reserved

import os
import sys
sys.path.append(os.getcwd())
import argparse
import numpy as np
import time
import torch.multiprocessing as multiprocessing
from nas_lib.utils.comm import random_id, setup_logger
from nas_lib.utils.utils_darts import compute_best_test_losses, compute_darts_test_losses
from nas_lib.algos_darts.build_open_algos import build_open_algos
from nas_lib.data.darts import DataSetDarts
from nas_lib.params_nas import algo_params_open_domain
from configs import ss_ccl_darts
import pickle


if __name__ == "__main__":
    loss_type = 'mae'
    parser = argparse.ArgumentParser(description='Args for darts search space.')
    parser.add_argument('--search_space', type=str, default='darts', help='darts')
    parser.add_argument('--gpus', type=int, default=1, help='Number of gpus')
    parser.add_argument('--algorithm', type=str, default='gin_predictor',
                        choices=['gin_predictor'], help='which parameters to use')
    parser.add_argument('--output_filename', type=str, default=random_id(64), help='name of output files')
    parser.add_argument('--node_nums', type=int, default=4, help='cell num')
    parser.add_argument('--log_level', type=str, default='DEBUG', help='information logging level')
    parser.add_argument('--seed', type=int, default=22, help='seed')
    parser.add_argument('--budget', type=int, default=100, help='searching budget.')
    parser.add_argument('--save_dir', type=str,
                        default='/home/aurora/data_disk_new/train_output_2021/darts_save_path/',
                        help='name of save directory')
    parser.add_argument('--self_supervised_algo', type=str, default='ss_ccl', choices=['ss_ccl', None],
                        help='The algorithm used to pre-train the model.')
    args = parser.parse_args()

    # make save directory
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(os.path.join(save_dir, 'model_pkl')):
        os.mkdir(os.path.join(save_dir, 'model_pkl'))
    if not os.path.exists(os.path.join(save_dir, 'results')):
        os.mkdir(os.path.join(save_dir, 'results'))
    if not os.path.exists(os.path.join(save_dir, 'pre_train_models')):
        os.mkdir(os.path.join(save_dir, 'pre_train_models'))
    # 2. build architecture training dataset
    arch_dataset = DataSetDarts()
    logger = setup_logger("nasbench_open_%s_cifar10" % args.search_space, args.save_dir, 0, log_level=args.log_level)
    algo_info = algo_params_open_domain(args.algorithm)
    algo_info['total_queries'] = args.budget
    starttime = time.time()
    multiprocessing.set_start_method('spawn')
    temp_k = 10
    file_name = save_dir + '/results/%s_%d.pkl' % (algo_info['algo_name'], algo_info['total_queries'])
    if args.self_supervised_algo:
        args.pre_train_path = ss_ccl_darts
    else:
        args.pre_train_path = None
    data = build_open_algos(algo_info['algo_name'])(search_space=arch_dataset,
                                                    algo_info=algo_info,
                                                    logger=logger,
                                                    gpus=args.gpus,
                                                    save_dir=save_dir,
                                                    seed=args.seed,
                                                    load_model=args.pre_train_path,
                                                    pre_train_algo=args.self_supervised_algo)
    if 'random' in algo_info['algo_name']:
        results, result_keys = compute_best_test_losses(data, temp_k, total_queries=algo_info['total_queries'])
        algo_result = np.round(results, 5)
    else:
        results, result_keys = compute_darts_test_losses(data, temp_k, total_queries=algo_info['total_queries'])
        algo_result = np.round(results, 5)
    print(algo_result)

    with open(file_name, 'wb') as f:
        pickle.dump(results, f)
        pickle.dump(result_keys, f)