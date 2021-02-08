# Copyright (c) Xidian University and Xi'an University of Posts & Telecommunications. All Rights Reserved

import sys


def supervised_ss_rl_params(param_str):
    if param_str == 'nasbench_101':
        params = {'lr': 5e-3, 'batch_size': 64, 'input_dim': 6}
        # params = {'lr': 5e-4, 'batch_size': 64, 'input_dim': 6}
    elif param_str == 'nasbench_201':
        params = {'lr': 1e-3, 'batch_size': 64, 'input_dim': 8}
        # params = {'lr': 1e-4, 'batch_size': 64, 'input_dim': 8}
    elif param_str == 'darts':
        params = {'lr': 5e-3, 'batch_size': 64, 'input_dim': 11}
    else:
        print('invalid params')
        sys.exit()
    return params


def unsupervised_ss_rl_params(param_str):
    if param_str == 'nasbench_101':
        params = {'lr': 5e-4, 'batch_size': 64, 'input_dim': 6}
    elif param_str == 'nasbench_201':
        params = {'lr': 1e-3, 'batch_size': 64, 'input_dim': 8}
    elif param_str == 'darts':
        params = {'lr': 5e-4, 'batch_size': 64, 'input_dim': 11}
    else:
        print('invalid params')
        sys.exit()
    return params


def unsupervised_ss_rl_pre_train(param_str):
    if param_str == 'nasbench_101':
        params = {'lr': 5e-4, 'batch_size': 64, 'input_dim': 6}
    elif param_str == 'nasbench_201':
        params = {'lr': 1e-4, 'batch_size': 64, 'input_dim': 8}
    elif param_str == 'darts':
        params = {'lr': 5e-4, 'batch_size': 64, 'input_dim': 11}
    else:
        print('invalid params')
        sys.exit()
    return params


def unsupervised_ss_ccl_params(param_str):
    if param_str == 'nasbench_101':
        params = {'lr': 5e-3, 'batch_size': 64, 'input_dim': 6}
    elif param_str == 'nasbench_201':
        params = {'lr': 5e-3, 'batch_size': 64, 'input_dim': 8}
    elif param_str == 'darts':
        params = {'lr': 5e-3, 'batch_size': 64, 'input_dim': 11}
    else:
        print('invalid params')
        sys.exit()
    return params


# def unsupervised_ss_ccl_pre_train(param_str):
#     if param_str == 'nasbench_101':
#         params = {'lr': 5e-4, 'batch_size': 64, 'input_dim': 6}
#     elif param_str == 'nasbench_201':
#         params = {'lr': 1e-4, 'batch_size': 64, 'input_dim': 8}
#     elif param_str == 'darts':
#         params = {'lr': 5e-4, 'batch_size': 64, 'input_dim': 11}
#     else:
#         print('invalid params')
#         sys.exit()
#     return params


def brp_nas_params(predictor_type=None):
    params = {'num_features': 6,
              'num_layers': 4,
              'num_hidden': 600,
              'dropout_ratio': 0.2,
              'weight_init': 'thomas',
              'bias_init': 'thomas',
              'binary_classifier': True,
              'lr': 3.5e-4,
              'weight_decay': 5.0e-4,
              'lr_patience': 10,
              'es_patience': 35,
              'batch_size': 64,
              'shuffle': True,
              'optim_name': 'adamw',
              'lr_scheduler': 'cosine'}
    return params


def semi_nas_params(predictor_type=None):
    params = {'nodes': 7,
              'encoder_layers': 1,
              'hidden_size': 16,
              'mlp_layers': 2,
              'mlp_hidden_size': 64,
              'decoder_layers': 1,
              'source_length': 27,
              'encoder_length': 27,
              'decoder_length': 27,
              'dropout': 0.1,
              'l2_reg': 1e-4,
              'vocab_size': 7,
              'epochs': 200,
              'batch_size': 100,
              'lr': 0.001,
              'optimizer': 'adam',
              'grad_bound': 5.0,
              'iteration': 2,
              'trade_off': 0.8}
    return params


def bananas_params(args=None):
    if args.search_space == 'nasbench_101':
        params = {'loss': 'mae', 'num_layers': 4, 'layer_width': 512,
                  'epochs': 200, 'batch_size': 32, 'lr': .01, 'regularization': 0, 'verbose': 0, 'ensemble': 1}
    elif args.search_space == 'nasbench_201':
        params = {'loss': 'mae', 'num_layers': 10, 'layer_width': 200,
                  'epochs': 200, 'batch_size': 32, 'lr': .001, 'regularization': 0, 'verbose': 0, 'ensemble': 1}
    else:
        raise ValueError()
    return params


def bananas_p_params(args=None):
    # if args.search_space == 'nasbench_101':
    #     params = {'loss': 'mae', 'num_layers': 10, 'layer_width': 20,
    #               'epochs': 200, 'batch_size': 32, 'lr': .01, 'regularization': 1e-4, 'verbose': 0, 'ensemble': 1}
    # elif args.search_space == 'nasbench_201':
    #     params = {'loss': 'mae', 'num_layers': 10, 'layer_width': 20,
    #               'epochs': 200, 'batch_size': 32, 'lr': .001, 'regularization': 1e-4, 'verbose': 0, 'ensemble': 1}
    if args.search_space == 'nasbench_101':
        params = {'loss': 'mae', 'num_layers': 10, 'layer_width': 20,
                  'epochs': 200, 'batch_size': 32, 'lr': .01, 'regularization': 0, 'verbose': 0, 'ensemble': 1}
    elif args.search_space == 'nasbench_201':
        params = {'loss': 'mae', 'num_layers': 10, 'layer_width': 200,
                  'epochs': 200, 'batch_size': 32, 'lr': .001, 'regularization': 0, 'verbose': 0, 'ensemble': 1}
    else:
        raise ValueError()
    return params


def mlp_params(args=None):
    if args.search_space == 'nasbench_101':
        params = {'loss': 'mae', 'num_layers': 4, 'layer_width': (512, 2048, 2048, 512),
                  'epochs': 200, 'batch_size': 32, 'lr': .01, 'regularization': 0, 'verbose': 0, 'in_channel': 41}
    elif args.search_space == 'nasbench_201':
        params = {'loss': 'mae', 'num_layers': 4, 'layer_width': (512, 2048, 2048, 512),
                  'epochs': 200, 'batch_size': 32, 'lr': .001, 'regularization': 0, 'verbose': 0, 'in_channel': 58}
    else:
        raise ValueError()
    return params


def np_nas_params(args=None):
    if args.search_space == 'nasbench_101':
        params = {
            'lr': 1e-4,
            'epochs': 300,
            'input_dim': 6
        }
    elif args.search_space == 'nasbench_201':
        params = {
            'lr': 1e-4,
            'epochs': 300,
            'input_dim': 8
        }
    elif args.search_space == 'darts':
        params = {'lr': 1e-4, 'batch_size': 64, 'input_dim': 11}
    else:
        raise NotImplementedError('This search space does not support at present!')
    return params


def get_params(args, predictor_type, load_model=False, pre_train=False):
    if predictor_type == 'SS_RL' and load_model and not pre_train:
        return unsupervised_ss_rl_params(args.search_space)
    elif 'SS_RL' in predictor_type and load_model and not pre_train:
        return unsupervised_ss_rl_params(args.search_space)
    elif predictor_type == 'SS_RL' and not load_model and not pre_train:
        return supervised_ss_rl_params(args.search_space)
    elif predictor_type == 'SS_RL' and pre_train:
        return unsupervised_ss_rl_pre_train(args.search_space)
    # elif predictor_type == 'SS_CCL' and pre_train:
    #     return unsupervised_ss_ccl_pre_train(args.search_space)
    elif 'SS_CCL' in predictor_type and not pre_train:
        return unsupervised_ss_ccl_params(args.search_space)
    elif predictor_type == 'BRP_NAS':
        return brp_nas_params()
    elif predictor_type == 'SemiNAS':
        return semi_nas_params()
    elif predictor_type == 'BANANAS' or predictor_type == 'BANANAS_ADJ':
        return bananas_params(args)
    elif predictor_type == 'BANANAS_P':
        return bananas_p_params(args)
    elif predictor_type == 'NP_NAS':
        return np_nas_params(args)
    elif predictor_type == 'MLP':
        return mlp_params(args)
    elif predictor_type == 'ss_rl' and load_model and not pre_train:
        return unsupervised_ss_rl_params(args.search_space)
    elif predictor_type == 'ss_rl' and not load_model and not pre_train:
        return supervised_ss_rl_params(args.search_space)
    elif predictor_type == 'ss_rl' and pre_train:
        return unsupervised_ss_rl_pre_train(args.search_space)
    # elif predictor_type == 'SS_CCL' and pre_train:
    #     return unsupervised_ss_ccl_pre_train(args.search_space)
    elif 'ss_ccl' in predictor_type and not pre_train:
        return unsupervised_ss_ccl_params(args.search_space)
    elif predictor_type == 'brp_nas':
        return brp_nas_params()
    elif predictor_type == 'seminas':
        return semi_nas_params()
    elif predictor_type == 'bananas' or predictor_type == 'bananas_adj':
        return bananas_params(args)
    elif predictor_type == 'bananas_p':
        return bananas_p_params(args)
    elif predictor_type == 'np_nas':
        return np_nas_params(args)
    elif predictor_type == 'mlp':
        return mlp_params(args)
    else:
        raise NotImplementedError()