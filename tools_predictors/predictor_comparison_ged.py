import argparse
import os
import sys
sys.path.append(os.getcwd())
from nas_lib.data import data
from nas_lib.trainer.trainer_retrain import NASBenchReTrain
from nas_lib.utils.comm import set_random_seed
from collections import defaultdict
import os
import pickle
from nas_lib.utils.comm import random_id, random_id_int, setup_logger
from configs import nas_bench_101_all_data, darts_converted_with_label, ss_rl_nasbench_201, ss_rl_wo_ged_1000_nasbench_201
import math


def predictor_retrain_compare(args, predictor, train_data, test_data, flag, load_dir=None, train_epochs=None,
                              logger=None):
    retrainer = NASBenchReTrain(args, predictor,
                                len(train_data),
                                load_model=flag,
                                load_dir=load_dir,
                                train_epochs=train_epochs,
                                logger=logger)
    return retrainer.fit_g_data(train_data, test_data)


def main(args):
    file_name = 'log_%s_%d' % ('gpus', args.gpu)
    logger = setup_logger(file_name, args.save_dir, args.gpu, log_level='DEBUG',
                          filename='%s.txt' % file_name)
    logger.info(args)
    if args.search_space == 'nasbench_101':
        with open(nas_bench_101_all_data, 'rb') as fpkl:
            all_data = pickle.load(fpkl)
    elif args.search_space == 'nasbench_201':
        nasbench_datas = data.build_datasets(args)
        all_data = data.dataset_all(args, nasbench_datas)
    elif args.search_space == 'darts':
        with open(darts_converted_with_label, 'rb') as fb:
            all_data = pickle.load(fb)
    else:
        raise NotImplementedError(f'The search space {args.search_space} does not support now!')

    for k in range(args.trails):
        seed = random_id_int(4)
        set_random_seed(seed)
        s_results_dict = defaultdict(list)
        k_results_dict = defaultdict(list)
        logger.info(f'======================  Trails {k} Begin Setting Seed to {seed} ===========================')
        for budget in args.search_budget:
            train_data, test_data = data.dataset_split_idx(all_data, budget)
            print(f'budget: {budget}, train data size: {len(train_data)}, test data size: {len(test_data)}')
            for epochs in args.train_iterations:
                for predictor_type, dir in zip(args.predictor_list, args.load_dir):
                    logger.info(f'====  predictor type: {predictor_type}, load pretrain model True. '
                                f'Search budget is {budget}. Training epoch is {epochs}. '
                                f'The model save dir is {dir.split("/")[-1][:-3]}  ====')
                    spearman_corr, kendalltau_corr, duration = predictor_retrain_compare(args, predictor_type,
                                                                                         train_data, test_data,
                                                                                         flag=True,
                                                                                         load_dir=dir,
                                                                                         train_epochs=epochs,
                                                                                         logger=logger)
                    if math.isnan(spearman_corr):
                        spearman_corr = 0
                    if math.isnan(kendalltau_corr):
                        kendalltau_corr = 0
                    s_results_dict[predictor_type + '#' + str(budget) + '#' + str(epochs)].append(spearman_corr)
                    k_results_dict[predictor_type + '#' + str(budget) + '#' + str(epochs)].append(kendalltau_corr)
        file_id = random_id(6)
        save_path = os.path.join(args.save_dir, f'{file_id}_{args.predictor_list[0]}_{args.search_space.split("_")[-1]}_{args.gpu}_{k}.pkl')
        with open(save_path, 'wb') as fp:
            pickle.dump(s_results_dict, fp)
            pickle.dump(k_results_dict, fp)


if __name__ == '__main__':
    # Running this code have to modify the predictor_list, search_space, trails, seq_len, gpu, load_dir, save_dir
    parser = argparse.ArgumentParser(description='Predictor comparison parameters!')
    parser.add_argument('--compare_supervised', type=str, default='T')
    parser.add_argument('--predictor_list', default=['SS_RL_NORMALIZED',
                                                     'SS_RL_WO_NORMALIZED',
                                                     ],
                        nargs='+',
                        help='The analysis architecture dataset type!')
    parser.add_argument('--search_space', type=str, default='nasbench_201',
                        help='The search space.')
    parser.add_argument('--with_g_func', type=bool, default=False,
                        help='Using the g function after the backbone.')
    parser.add_argument('--trails', type=int, default=40, help='How many trails to carry out.')
    parser.add_argument('--seed', type=int, default=random_id_int(4), help='The seed value.')
    parser.add_argument('--dataname', type=str, default='cifar10-valid',
                        choices=['cifar10-valid', 'cifar10', 'cifar100', 'ImageNet16-120'],
                        help='The evaluation of dataset of NASBench-201.')
    parser.add_argument('--search_budget', type=list,
                        default=[20, 50, 100, 150, 200],
                        help='How many architectures are selected to train the neural predictor.')
    parser.add_argument('--train_iterations', type=list,
                        default=[50, 100, 150, 200, 250, 300],
                        help='How many training iterations are used to train the model.')
    parser.add_argument('--gpu', type=int, default=1, help='Choose which gpu to train the neural network.')
    parser.add_argument('--load_dir',  nargs='+',
                        default=[],
                        help='The pre-trained unsupervised model save path.')
    parser.add_argument('--save_dir', type=str,
                        default='/home/aurora/data_disk_new/train_output_2021/darts_save_path/',
                        help='The pre-trained unsupervised model save path.')
    args = parser.parse_args()

    if args.search_space == 'nasbench_201':
        args.seq_len = 96
        if not args.load_dir:
            args.load_dir = [ss_rl_nasbench_201, ss_rl_wo_ged_1000_nasbench_201]
    else:
        raise NotImplementedError('This search space does not support at present!')

    main(args)