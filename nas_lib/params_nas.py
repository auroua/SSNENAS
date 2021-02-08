import sys


def meta_neuralnet_params(param_str):
    if param_str == 'nasbench_101':
        params = {'search_space': 'nasbench_101', 'loss': 'mae', 'num_layers': 10, 'layer_width': 20,
                  'epochs': 150, 'batch_size': 32, 'lr': .01, 'regularization': 0, 'verbose': 0}
    elif param_str == 'nasbench_201':
        params = {'search_space': 'nasbench_201', 'loss': 'mae', 'num_layers': 10, 'layer_width': 200,
                  'epochs': 200, 'batch_size': 32, 'lr': .001, 'regularization': 0, 'verbose': 0}
    else:
        print('invalid meta neural net params')
        sys.exit()
    return params


def algo_params_close_domain(param_str, search_budget=100, dataname='cifar10'):
    """
      Return params list based on param_str.
      These are the parameters used to produce the figures in the paper
      For AlphaX and Reinforcement Learning, we used the corresponding github repos:
      https://github.com/linnanwang/AlphaX-NASBench101
      https://github.com/automl/nas_benchmarks
    """
    params = []

    if dataname == 'cifar10-valid':
        rate = 10.
    elif dataname == 'cifar100':
        rate = 30.
    elif dataname == 'ImageNet16-120':
        rate = 55
    else:
        raise NotImplementedError()

    if param_str == 'nasbench_101':
        params.append({'algo_name': 'random', 'total_queries': search_budget})
        params.append({'algo_name': 'evolution', 'total_queries': search_budget, 'population_size': 30, 'num_init': 10, 'k': 10,
                       'tournament_size': 10, 'mutation_rate': 1.0})
        params.append({'algo_name': 'bananas', 'total_queries': search_budget, 'num_ensemble': 5, 'allow_isomorphisms': False,
                       'acq_opt_type': 'mutation', 'candidate_nums': 100, 'num_init': 10, 'k': 10,
                       'encode_paths': True})
        params.append({'algo_name': 'bananas_f', 'total_queries': search_budget, 'num_ensemble': 5,
                       'allow_isomorphisms': False, 'acq_opt_type': 'mutation', 'candidate_nums': 100, 'num_init': 10,
                       'k': 10, 'encode_paths': False})
        params.append({'algo_name': 'bananas_context', 'total_queries': search_budget, 'num_ensemble': 5,
                       'allow_isomorphisms': False, 'acq_opt_type': 'mutation', 'candidate_nums': 100, 'num_init': 10,
                       'k': 10, 'encode_paths': False})
        params.append({'algo_name': 'gin_predictor', 'total_queries': search_budget, 'agent': 'gin_predictor',
                       'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'candidate_nums': 100, 'epochs': 300})
        params.append({'algo_name': 'gin_predictor_fixed_nums', 'total_queries': search_budget, 'agent': 'gin_predictor',
                       'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'candidate_nums': 100, 'epochs': 300,
                       'training_nums': 50})
        params.append({'algo_name': 'gin_predictor_ss_rl', 'total_queries': search_budget, 'k': 10,
                       'agent': 'gin_predictor', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005,
                       'candidate_nums': 100, 'epochs': 150, 'predictor_type': 'ss_rl'})
        params.append({'algo_name': 'gin_predictor_ss_rl_num_fixed', 'total_queries': search_budget, 'k': 10,
                       'agent': 'gin_predictor', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005,
                       'candidate_nums': 100, 'epochs': 150, 'predictor_type': 'ss_rl', 'training_nums': 90})
        params.append({'algo_name': 'gin_predictor_ss_ccl', 'total_queries': search_budget, 'k': 10,
                       'agent': 'gin_predictor', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005,
                       'candidate_nums': 100, 'epochs': 200, 'predictor_type': 'ss_ccl'})
        params.append({'algo_name': 'gin_predictor_ss_ccl_num_fixed', 'total_queries': search_budget, 'k': 10,
                       'agent': 'gin_predictor', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005,
                       'candidate_nums': 100, 'epochs': 200, 'predictor_type': 'ss_ccl', 'training_nums': 90})
    elif param_str == 'nasbench_201':
        params.append({'algo_name': 'random', 'total_queries': search_budget})
        params.append({'algo_name': 'evolution', 'total_queries': search_budget, 'population_size': 30, 'num_init': 10,
                       'k': 10, 'tournament_size': 10, 'mutation_rate': 1.0, 'allow_isomorphisms': False,
                       'deterministic': True})
        params.append({'algo_name': 'bananas', 'total_queries': search_budget, 'num_ensemble': 5,
                       'allow_isomorphisms': False, 'acq_opt_type': 'mutation', 'candidate_nums': 100, 'num_init': 10,
                       'k': 10, 'encode_paths': True, 'eva_new': False})
        params.append({'algo_name': 'bananas_f', 'total_queries': search_budget, 'num_ensemble': 5,
                       'allow_isomorphisms': False, 'acq_opt_type': 'mutation', 'candidate_nums': 100, 'num_init': 10,
                       'k': 10, 'encode_paths': False})
        params.append({'algo_name': 'bananas_context', 'total_queries': search_budget, 'num_ensemble': 5,
                       'allow_isomorphisms': False, 'acq_opt_type': 'mutation', 'candidate_nums': 100, 'num_init': 10,
                       'k': 10, 'encode_paths': False})
        params.append({'algo_name': 'gin_predictor', 'total_queries': search_budget, 'agent': 'gin_predictor',
                       'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'candidate_nums': 100, 'epochs': 300,
                       'rate': rate})
        params.append({'algo_name': 'gin_predictor_fixed_nums', 'total_queries': search_budget, 'agent': 'gin_predictor',
                       'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'candidate_nums': 100, 'epochs': 300,
                       'training_nums': 50, 'rate': rate})
        params.append({'algo_name': 'gin_predictor_ss_rl', 'total_queries': search_budget, 'k': 10,
                       'agent': 'gin_predictor', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005,
                       'candidate_nums': 100, 'epochs': 150, 'predictor_type': 'ss_rl', 'rate': rate})
        params.append({'algo_name': 'gin_predictor_ss_rl_num_fixed', 'total_queries': search_budget, 'k': 10,
                       'agent': 'gin_predictor', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'rate': rate,
                       'candidate_nums': 100, 'epochs': 150, 'predictor_type': 'ss_rl', 'training_nums': 50})
        params.append({'algo_name': 'gin_predictor_ss_ccl', 'total_queries': search_budget, 'k': 10,
                       'agent': 'gin_predictor', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'rate': rate,
                       'candidate_nums': 100, 'epochs': 150, 'predictor_type': 'ss_ccl'})
        params.append({'algo_name': 'gin_predictor_ss_ccl_num_fixed', 'total_queries': search_budget, 'k': 10,
                       'agent': 'gin_predictor', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'rate': rate,
                       'candidate_nums': 100, 'epochs': 150, 'predictor_type': 'ss_ccl', 'training_nums': 50})
    elif param_str == 'nasbench_101_fixed':
        params.append({'algo_name': 'gin_predictor_ss_rl_num_fixed', 'total_queries': search_budget, 'k': 10,
                       'agent': 'gin_predictor', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005,
                       'candidate_nums': 100, 'epochs': 150, 'predictor_type': 'ss_rl', 'training_nums': 20})
        params.append({'algo_name': 'gin_predictor_ss_rl_num_fixed', 'total_queries': search_budget, 'k': 10,
                       'agent': 'gin_predictor', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005,
                       'candidate_nums': 100, 'epochs': 150, 'predictor_type': 'ss_rl', 'training_nums': 50})
        params.append({'algo_name': 'gin_predictor_ss_rl_num_fixed', 'total_queries': search_budget, 'k': 10,
                       'agent': 'gin_predictor', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005,
                       'candidate_nums': 100, 'epochs': 150, 'predictor_type': 'ss_rl', 'training_nums': 80})
        params.append({'algo_name': 'gin_predictor_ss_rl_num_fixed', 'total_queries': search_budget, 'k': 10,
                       'agent': 'gin_predictor', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005,
                       'candidate_nums': 100, 'epochs': 150, 'predictor_type': 'ss_rl', 'training_nums': 110})
        params.append({'algo_name': 'gin_predictor_ss_rl_num_fixed', 'total_queries': search_budget, 'k': 10,
                       'agent': 'gin_predictor', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005,
                       'candidate_nums': 100, 'epochs': 150, 'predictor_type': 'ss_rl', 'training_nums': 150})

        params.append({'algo_name': 'gin_predictor_ss_ccl_num_fixed', 'total_queries': search_budget, 'k': 10,
                       'agent': 'gin_predictor', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005,
                       'candidate_nums': 100, 'epochs': 200, 'predictor_type': 'ss_ccl', 'training_nums': 20})
        params.append({'algo_name': 'gin_predictor_ss_ccl_num_fixed', 'total_queries': search_budget, 'k': 10,
                       'agent': 'gin_predictor', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005,
                       'candidate_nums': 100, 'epochs': 200, 'predictor_type': 'ss_ccl', 'training_nums': 50})
        params.append({'algo_name': 'gin_predictor_ss_ccl_num_fixed', 'total_queries': search_budget, 'k': 10,
                       'agent': 'gin_predictor', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005,
                       'candidate_nums': 100, 'epochs': 200, 'predictor_type': 'ss_ccl', 'training_nums': 80})
        params.append({'algo_name': 'gin_predictor_ss_ccl_num_fixed', 'total_queries': search_budget, 'k': 10,
                       'agent': 'gin_predictor', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005,
                       'candidate_nums': 100, 'epochs': 200, 'predictor_type': 'ss_ccl', 'training_nums': 110})
        params.append({'algo_name': 'gin_predictor_ss_ccl_num_fixed', 'total_queries': search_budget, 'k': 10,
                       'agent': 'gin_predictor', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005,
                       'candidate_nums': 100, 'epochs': 200, 'predictor_type': 'ss_ccl', 'training_nums': 150})
    elif param_str == 'nasbench_201_fixed':
        params.append({'algo_name': 'gin_predictor_ss_rl_num_fixed', 'total_queries': search_budget, 'k': 10,
                       'agent': 'gin_predictor', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'rate': rate,
                       'candidate_nums': 100, 'epochs': 150, 'predictor_type': 'ss_rl', 'training_nums': 20})
        params.append({'algo_name': 'gin_predictor_ss_rl_num_fixed', 'total_queries': search_budget, 'k': 10,
                       'agent': 'gin_predictor', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'rate': rate,
                       'candidate_nums': 100, 'epochs': 150, 'predictor_type': 'ss_rl', 'training_nums': 50})
        params.append({'algo_name': 'gin_predictor_ss_rl_num_fixed', 'total_queries': search_budget, 'k': 10,
                       'agent': 'gin_predictor', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'rate': rate,
                       'candidate_nums': 100, 'epochs': 150, 'predictor_type': 'ss_rl', 'training_nums': 80})
        params.append({'algo_name': 'gin_predictor_ss_rl_num_fixed', 'total_queries': search_budget, 'k': 10,
                       'agent': 'gin_predictor', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'rate': rate,
                       'candidate_nums': 100, 'epochs': 150, 'predictor_type': 'ss_rl', 'training_nums': 100})

        params.append({'algo_name': 'gin_predictor_ss_ccl_num_fixed', 'total_queries': search_budget, 'k': 10,
                       'agent': 'gin_predictor', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'rate': rate,
                       'candidate_nums': 100, 'epochs': 150, 'predictor_type': 'ss_ccl', 'training_nums': 20})
        params.append({'algo_name': 'gin_predictor_ss_ccl_num_fixed', 'total_queries': search_budget, 'k': 10,
                       'agent': 'gin_predictor', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'rate': rate,
                       'candidate_nums': 100, 'epochs': 150, 'predictor_type': 'ss_ccl', 'training_nums': 50})
        params.append({'algo_name': 'gin_predictor_ss_ccl_num_fixed', 'total_queries': search_budget, 'k': 10,
                       'agent': 'gin_predictor', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'rate': rate,
                       'candidate_nums': 100, 'epochs': 150, 'predictor_type': 'ss_ccl', 'training_nums': 80})
        params.append({'algo_name': 'gin_predictor_ss_ccl_num_fixed', 'total_queries': search_budget, 'k': 10,
                       'agent': 'gin_predictor', 'num_init': 10, 'allow_isomorphisms': False, 'lr': 0.005, 'rate': rate,
                       'candidate_nums': 100, 'epochs': 150, 'predictor_type': 'ss_ccl', 'training_nums': 100})
    elif param_str == 'experiment':
        pass
    else:
        print('invalid algorithm params')
        sys.exit()

    print('* Running experiment: ' + param_str)
    return params


def algo_params_open_domain(param_str):
    if param_str == 'gin_predictor':   # gin_predictor   gin_predictor_fixed_num
        param = {'algo_name': 'gin_predictor_fixed_num', 'total_queries': 30, 'agent': 'gin_predictor', 'num_init': 10,
                 'allow_isomorphisms': False, 'k': 10, 'epochs': 300, 'batch_size': 32, 'lr': 0.005,
                 'encode_path': True, 'candidate_nums': 100, 'mutate_rate': 2.0, 'filter_method': 'pape',
                 'fixed_num': 70}
    else:
        raise NotImplementedError("This algorithm have not implement!")
    print('* Running experiment: ' + str(param))
    return param