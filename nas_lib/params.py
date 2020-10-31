# Copyright (c) Xidian University and Xi'an University of Posts & Telecommunications. All Rights Reserved

import sys


def unsupervised_ss_rl_params(param_str):
    if param_str == 'nasbench_101':
        params = {'lr': 5e-4, 'batch_size': 64, 'input_dim': 6}
    elif param_str == 'nasbench_201':
        params = {'lr': 1e-4, 'batch_size': 64, 'input_dim': 8}
    else:
        print('invalid params')
        sys.exit()
    return params


def unsupervised_ss_ccl_params(param_str):
    if param_str == 'nasbench_101':
        params = {'lr': 5e-4, 'batch_size': 64, 'input_dim': 6}
    elif param_str == 'nasbench_201':
        params = {'lr': 1e-4, 'batch_size': 64, 'input_dim': 8}
    else:
        print('invalid params')
        sys.exit()
    return params


def get_params(args, predictor_type):
    if predictor_type == 'SS_RL':
        return unsupervised_ss_rl_params(args.search_space)
    elif 'SS_CCL' in predictor_type:
        return unsupervised_ss_ccl_params(args.search_space)
    else:
        raise NotImplementedError()