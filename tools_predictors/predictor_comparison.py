import argparse
import os
import sys
sys.path.append(os.getcwd())
# os.environ["CUDA_DEVICE_ORDER"] = "0000:01:00.0"   # gpu 0
# os.environ["CUDA_DEVICE_ORDER"]="0000:02:00.0"
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from nas_lib.data import data
from nas_lib.trainer.trainer_retrain import NASBenchReTrain
from nas_lib.utils.comm import set_random_seed
from collections import defaultdict
import os
import pickle
from nas_lib.utils.comm import random_id, random_id_int, setup_logger
from configs import nas_bench_101_all_data
from configs import ss_rl_nasbench_101, ss_ccl_nasbench_101, darts_converted_with_label
import tensorflow as tf
import math


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)


def predictor_comparision(args, predictor, train_data, test_data, flag, load_dir=None, train_epochs=None,
                          logger=None):
    args.with_g_func = False
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
        duration_dict = defaultdict(list)
        logger.info(f'======================  Trails {k} Begin Setting Seed to {seed} ===========================')
        for budget in args.search_budget:
            train_data, test_data = data.dataset_split_idx_predictive_comparison(all_data, budget)
            print(f'budget: {budget}, train data size: {len(train_data)}, test data size: {len(test_data)}')
            if args.compare_supervised == 'T':
                logger.info(f'====  predictor type: SUPERVISED, load pretrain model False, '
                            f'search budget is {budget}. Training epoch is {args.epochs} ====')
                spearman_corr, kendalltau_corr, duration = predictor_comparision(args, 'SS_RL',
                                                                                 train_data, test_data,
                                                                                 flag=False,
                                                                                 train_epochs=args.epochs,
                                                                                 logger=logger)
                if math.isnan(spearman_corr):
                    spearman_corr = 0
                if math.isnan(kendalltau_corr):
                    kendalltau_corr = 0
                s_results_dict[f'supervised#{budget}#{args.epochs}'].append(spearman_corr)
                k_results_dict[f'supervised#{budget}#{args.epochs}'].append(kendalltau_corr)
                duration_dict[f'supervised#{budget}#{args.epochs}'].append(duration)
            for predictor_type, dir in zip(args.predictor_list, args.load_dir):
                logger.info(f'====  predictor type: {predictor_type}, load pretrain model True. '
                            f'Search budget is {budget}. Training epoch is {args.epochs}. '
                            f'The model save dir is {dir.split("/")[-1][:-3]}  ====')
                spearman_corr, kendalltau_corr, duration = predictor_comparision(args, predictor_type,
                                                                                 train_data, test_data,
                                                                                 flag=True,
                                                                                 load_dir=dir,
                                                                                 train_epochs=args.epochs,
                                                                                 logger=logger)
                if math.isnan(spearman_corr):
                    spearman_corr = 0
                if math.isnan(kendalltau_corr):
                    kendalltau_corr = 0
                s_results_dict[predictor_type + '#' + str(budget) + '#' + str(args.epochs)].append(spearman_corr)
                k_results_dict[predictor_type + '#' + str(budget) + '#' + str(args.epochs)].append(kendalltau_corr)
                duration_dict[predictor_type + '#' + str(budget) + '#' + str(args.epochs)].append(duration)
        file_id = random_id(6)
        save_path = os.path.join(args.save_dir, f'{file_id}_{args.predictor_list[0]}_{args.search_space.split("_")[-1]}_{args.gpu}_{k}.pkl')
        with open(save_path, 'wb') as fp:
            pickle.dump(s_results_dict, fp)
            pickle.dump(k_results_dict, fp)
            pickle.dump(duration_dict, fp)


if __name__ == '__main__':
    # Running this code have to modify the predictor_list, search_space, trails, seq_len, gpu, load_dir, save_dir
    parser = argparse.ArgumentParser(description='Predictive performance comparison!')
    parser.add_argument('--compare_supervised', type=str, default='T')
    parser.add_argument('--predictor_list', default='', nargs='+',
                        help='predictor types!')
    parser.add_argument('--search_space', type=str, default='nasbench_101',
                        choices=['nasbench_101'],
                        help='The search space.')
    parser.add_argument('--trails', type=int, default=40, help='How many trails to carry out.')
    parser.add_argument('--epochs', type=int, default=200, help='How many epochs to train neural predictor.')
    parser.add_argument('--seed', type=int, default=random_id_int(4), help='The seed value.')
    parser.add_argument('--dataname', type=str, default='cifar100',
                        choices=['cifar10-valid', 'cifar10', 'cifar100', 'ImageNet16-120'],
                        help='The evaluation of dataset of NASBench-201.')
    parser.add_argument('--search_budget', type=list,
                        default=[20, 50, 100, 200],
                        help='The percents of dataset used to train neural architectures.')
    parser.add_argument('--gpu', type=int, default=0, help='Choose which gpu to train the neural network.')
    parser.add_argument('--load_dir',  nargs='+',
                        default=[],
                        help='The pre-trained unsupervised model save path.')
    parser.add_argument('--save_dir', type=str,
                        default='/home/aurora/data_disk_new/train_output_2021/darts_save_path/',
                        help='The pre-trained unsupervised model save path.')
    args = parser.parse_args()

    if args.search_space == 'nasbench_101':
        args.seq_len = 120
        if not args.load_dir:
            args.load_dir = [ss_rl_nasbench_101, ss_ccl_nasbench_101, '', '', '', '', '']
            args.predictor_list = ['SS_RL', 'SS_CCL', 'SemiNAS', 'NP_NAS', 'MLP']
    else:
        raise NotImplementedError('This search space does not support at present!')

    main(args)