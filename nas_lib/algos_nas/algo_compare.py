# Copyright (c) XiDian University and Xi'an University of Posts&Telecommunication. All Rights Reserved

import copy
import sys
import numpy as np
from .random import random_search_nasbench_101, random_search_nasbench_201
from .evolution import evolution_search_nasbench_101, evolution_search_nasbench_201
from .bananas import bananas_nasbench_101, bananas_nasbench_201
from .predictor import gin_predictor_nasbench_101, gin_predictor_nasbench_201, \
    gin_predictor_train_num_restract_nasbench_101, gin_predictor_train_num_restract_nasbench_201
from .predictor_unsupervised import gin_unsupervised_predictor, gin_unsupervised_predictor_fix_num


def run_nas_algos_nasbench_101(algo_params, metann_params, search_space, gpu=None, logger=None, with_details='F',
                               model_dir=None):
    mp = copy.deepcopy(metann_params)
    ps = copy.deepcopy(algo_params)
    algo_name = ps.pop('algo_name')
    if algo_name == 'random':
        data = random_search_nasbench_101(search_space, logger=logger, **ps)
    elif algo_name == 'evolution':
        data = evolution_search_nasbench_101(search_space, logger=logger, **ps)
    elif algo_name == 'bananas':
        mp.pop('search_space')
        data = bananas_nasbench_101(search_space, mp, gpu=gpu, logger=logger, algo_name=algo_name, **ps)
    elif algo_name == 'bananas_f':
        mp.pop('search_space')
        data = bananas_nasbench_101(search_space, mp, gpu=gpu, logger=logger, algo_name=algo_name, **ps)
    elif algo_name == 'bananas_context':
        mp.pop('search_space')
        data = bananas_nasbench_101(search_space, mp, gpu=gpu, logger=logger, algo_name=algo_name, **ps)
    elif algo_name == 'gin_predictor':
        ps['algo_name'] = algo_name
        data = gin_predictor_nasbench_101(search_space, gpu=gpu, logger=logger, **ps)
    elif algo_name == 'gin_predictor_fixed_nums':
        ps['algo_name'] = algo_name
        data = gin_predictor_train_num_restract_nasbench_101(search_space, gpu=gpu, logger=logger, **ps)
    elif algo_name == 'gin_predictor_ss_rl':
        ps['algo_name'] = algo_name
        data = gin_unsupervised_predictor(search_space, gpu=gpu, logger=logger, model_dir=model_dir,
                                          benchmark='nasbench_101', **ps)
    elif algo_name == 'gin_predictor_ss_rl_num_fixed':
        ps['algo_name'] = algo_name
        data = gin_unsupervised_predictor_fix_num(search_space, gpu=gpu, logger=logger, benchmark='nasbench_101',
                                                  model_dir=model_dir, **ps)
    elif algo_name == 'gin_predictor_ss_ccl':
        ps['algo_name'] = algo_name
        data = gin_unsupervised_predictor(search_space, gpu=gpu, logger=logger, model_dir=model_dir,
                                          benchmark='nasbench_101', **ps)
    elif algo_name == 'gin_predictor_ss_ccl_num_fixed':
        ps['algo_name'] = algo_name
        data = gin_unsupervised_predictor_fix_num(search_space, gpu=gpu, logger=logger, benchmark='nasbench_101',
                                                  model_dir=model_dir, **ps)
    else:
        print('invalid algorithm name')
        sys.exit()
    k = 10
    if 'k' in ps:
        k = ps['k']
    if with_details == 'T':
        return compute_best_test_losses_with_details(data, k, ps['total_queries'])
    else:
        return compute_best_test_losses(data, k, ps['total_queries'])


def run_nas_algos_nasbench_201(algo_params, metann_params, search_space, gpu=None, logger=None, model_dir=None):
    mp = copy.deepcopy(metann_params)
    ps = copy.deepcopy(algo_params)
    algo_name = ps.pop('algo_name')
    if algo_name == 'random':
        data = random_search_nasbench_201(search_space, logger=logger, **ps)
    elif algo_name == 'evolution':
        data = evolution_search_nasbench_201(search_space, logger=logger, **ps)
    elif algo_name == 'bananas':
        mp.pop('search_space')
        data = bananas_nasbench_201(search_space, mp, gpu=gpu, logger=logger, algo_name=algo_name, **ps)
    elif algo_name == 'bananas_f':
        mp.pop('search_space')
        mp['layer_width'] = 20
        data = bananas_nasbench_201(search_space, mp, gpu=gpu, logger=logger, algo_name=algo_name, **ps)
    elif algo_name == 'bananas_context':
        mp.pop('search_space')
        mp['layer_width'] = 20
        data = bananas_nasbench_201(search_space, mp, gpu=gpu, logger=logger, algo_name=algo_name, **ps)
    elif algo_name == 'gin_predictor':
        ps['algo_name'] = algo_name
        data = gin_predictor_nasbench_201(search_space, gpu=gpu, logger=logger, **ps)
    elif algo_name == 'gin_predictor_fixed_nums':
        ps['algo_name'] = algo_name
        data = gin_predictor_train_num_restract_nasbench_201(search_space, gpu=gpu, logger=logger, **ps)
    elif algo_name == 'gin_predictor_ss_rl':
        ps['algo_name'] = algo_name
        data = gin_unsupervised_predictor(search_space, gpu=gpu, logger=logger, model_dir=model_dir,
                                          benchmark='nasbench_201', **ps)
    elif algo_name == 'gin_predictor_ss_rl_num_fixed':
        ps['algo_name'] = algo_name
        data = gin_unsupervised_predictor_fix_num(search_space, gpu=gpu, logger=logger, benchmark='nasbench_201',
                                                  model_dir=model_dir, **ps)
    elif algo_name == 'gin_predictor_ss_ccl':
        ps['algo_name'] = algo_name
        data = gin_unsupervised_predictor(search_space, gpu=gpu, logger=logger, benchmark='nasbench_201',
                                          model_dir=model_dir, **ps)
    elif algo_name == 'gin_predictor_ss_ccl_num_fixed':
        ps['algo_name'] = algo_name
        data = gin_unsupervised_predictor_fix_num(search_space, gpu=gpu, logger=logger, benchmark='nasbench_201',
                                                  model_dir=model_dir, **ps)
    else:
        print('invalid algorithm name')
        sys.exit()
    k = 10
    if 'k' in ps:
        k = ps['k']
    return compute_best_test_losses(data, k, ps['total_queries'])


def compute_best_test_losses(data, k, total_queries):
    """
    Given full data from a completed nas algorithm,
    output the test error of the arch with the best val error
    after every multiple of k
    """
    results = []
    for query in range(k, total_queries + k, k):
        best_arch = sorted(data[:query], key=lambda i: i[4])[0]
        test_error = best_arch[5]
        results.append((query, test_error))
    return results


def compute_best_test_losses_with_details(data, k, total_queries):
    """
    Given full data from a completed nas algorithm,
    output the test error of the arch with the best val error
    after every multiple of k
    """
    results = []
    for query in range(k, total_queries + k, k):
        best_arch = sorted(data[:query], key=lambda i: i[4])[0]
        test_error = best_arch[5]
        results.append((query, test_error))
    val_distribution = [d[4] for d in data]
    val_datas_np = np.array(val_distribution).reshape((-1, 10))
    test_distribution = [d[5] for d in data]
    test_datas_np = np.array(test_distribution).reshape((-1, 10))
    dist_results = {'val': val_datas_np,
                    'test': test_datas_np}
    return results, dist_results