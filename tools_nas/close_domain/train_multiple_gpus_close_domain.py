# Copyright (c) Xidian University and Xi'an University of Posts & Telecommunications. All Rights Reserved


import argparse
import os
import sys
sys.path.append(os.getcwd())
from nas_lib.algos_nas.algo_compare import run_nas_algos_nasbench_101, run_nas_algos_nasbench_201
from nas_lib.params_nas import meta_neuralnet_params
from nas_lib.params_nas import algo_params_close_domain as algo_params
from nas_lib.data.data import build_datasets
import tensorflow as tf
import psutil
from nas_lib.utils.comm import set_random_seed, random_id, setup_logger
import torch.multiprocessing as multiprocessing
from torch.multiprocessing import Process
from torch.multiprocessing import Queue
import pickle
import numpy as np
import time
from configs import ss_rl_nasbench_101, ss_rl_nasbench_201, ss_ccl_nasbench_101, ss_ccl_nasbench_201


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)


def ansyc_multiple_process_train(args, save_dir):
    q = Queue(10)
    data_lists = [build_datasets(args) for _ in range(args.gpus)]

    p_producer = Process(target=data_producers, args=(args, q))
    p_consumers = [Process(target=data_consumers, args=(args, q, save_dir, i, data_lists[i])) for i in range(args.gpus)]

    p_producer.start()
    for p in p_consumers:
        p.start()

    p_producer.join()
    for p in p_consumers:
        p.join()


def data_producers(args, queue):
    trials = args.trials
    for i in range(trials):
        queue.put({
            'iterate': i
        })
    for _ in range(args.gpus):
        queue.put('done')


def data_consumers(args, q, save_dir, i, search_space):
    set_random_seed(int(str(time.time()).split('.')[0][::-1][:9]))
    file_name = 'log_%s_%d' % ('gpus', i)
    logger = setup_logger(file_name, save_dir, i, log_level='DEBUG',
                          filename='%s.txt' % file_name)
    while True:
        msg = q.get()
        if msg == 'done':
            logger.info('thread %d end' % i)
            break
        iterations = msg['iterate']
        run_experiments_bananas_paradigm(args, save_dir, i, iterations, logger, search_space)


def run_experiments_bananas_paradigm(args, save_dir, i, iterations, logger, search_space):
    out_file = args.output_filename + '_gpus_%d_' % i + 'iter_%d' % iterations
    metann_params = meta_neuralnet_params(args.search_space)
    algorithm_params = algo_params(args.algo_params, args.search_budget, args.dataname)
    num_algos = len(algorithm_params)
    results = []
    result_dist = []
    walltimes = []
    for j in range(num_algos):
        logger.info(' * Running algorithm: {}'.format(algorithm_params[j]))
        logger.info(' * Loss type: {}'.format(args.loss_type))
        logger.info(' * Trials: {}, Free Memory available {}'.format(iterations,
                                                                     psutil.virtual_memory().free/(1024*1024)))
        starttime = time.time()
        if args.algo_params == 'nasbench_101' or args.algo_params == 'nasbench_101_fixed':
            model_dir_dict_101 = {'ss_rl': ss_rl_nasbench_101,
                                  'ss_ccl': ss_ccl_nasbench_101}
            algo_result = run_nas_algos_nasbench_101(algorithm_params[j], metann_params, search_space, gpu=i,
                                                     logger=logger, with_details=args.with_details,
                                                     model_dir=model_dir_dict_101)
        elif args.algo_params == 'nasbench_201' or args.algo_params == 'nasbench_201_fixed':
            model_dir_dict_201 = {'ss_rl': ss_rl_nasbench_201,
                                  'ss_ccl': ss_ccl_nasbench_201}
            algo_result = run_nas_algos_nasbench_201(algorithm_params[j], metann_params, search_space, gpu=i,
                                                     logger=logger, model_dir=model_dir_dict_201)
        else:
            raise NotImplementedError("This algorithm does not support!")

        algo_result = np.round(algo_result, 5)
        # add walltime and results
        walltimes.append(time.time() - starttime)
        results.append(algo_result)

    filename = os.path.join(save_dir, '{}_{}.pkl'.format(out_file, i))
    logger.info(' * Trial summary: (params, results, walltimes)')
    logger.info(algorithm_params)
    logger.info(metann_params)
    for k in range(results[0].shape[0]):
        length = len(results)
        results_line = []
        for j in range(length):
            if j == 0:
                results_line.append(int(results[j][k, 0]))
                results_line.append(results[j][k, 1])
            else:
                results_line.append(results[j][k, 1])
        results_str = '  '.join([str(k) for k in results_line])
        logger.info(results_str)
    logger.info(walltimes)
    logger.info(' * Saving to file {}'.format(filename))
    with open(filename, 'wb') as f:
        if args.with_details == 'T':
            pickle.dump([algorithm_params, metann_params, results, result_dist, walltimes], f)
        else:
            pickle.dump([algorithm_params, metann_params, results, walltimes], f)
        f.close()
    logger.info('#######################################################  Trails %d End  '
                '#######################################################' % iterations)


def main(args):
    save_dir = args.save_dir
    if not save_dir:
        save_dir = args.algo_params + '/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    multiprocessing.set_start_method('spawn')
    ansyc_multiple_process_train(args, save_dir)


if __name__ == "__main__":
    # change from nasbench_101 to nasbench_201 have to modify four parameters: search_budget, search_space,
    # algo_params, model_dir.
    parser = argparse.ArgumentParser(description='Args for algorithms compare.')
    parser.add_argument('--trials', type=int, default=600, help='Number of trials')
    parser.add_argument('--search_space', type=str, default='nasbench_101',
                        choices=['nasbench_101', 'nasbench_201'],
                        help='The nasbench search space.')
    parser.add_argument('--dataname', type=str, default='cifar100',
                        choices=['cifar10-valid', 'cifar10', 'cifar100', 'ImageNet16-120'],
                        help='The evaluation of dataset of NASBench-201.')
    parser.add_argument('--algo_params', type=str, default='nasbench_101_fixed',
                        choices=['nasbench_101', 'nasbench_201', 'nasbench_101_fixed', 'nasbench_201_fixed'],
                        help='which algorithms to compare')
    parser.add_argument('--output_filename', type=str, default=random_id(64), help='name of output files')
    parser.add_argument('--gpus', type=int, default=2, help='The num of gpus used for search.')
    parser.add_argument('--loss_type', type=str, default="mae", help='Loss used to train architecture.')
    parser.add_argument('--with_details', type=str, default="F", help='Record detailed training procedure.')
    parser.add_argument('--save_dir', type=str,
                        default='/home/aurora/data_disk_new/train_output_2021/predictor_performance_comparision/npenas_nasbench_101_ss_ccl_batch_size_compare/',
                        help='output directory')
    args = parser.parse_args()

    if args.search_space == 'nasbench_101':
        args.search_budget = 150
    elif args.search_space == 'nasbench_201':
        args.search_budget = 100
    else:
        raise NotImplementedError('Not implement!')
    main(args)
