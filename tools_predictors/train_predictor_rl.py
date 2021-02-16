import argparse
import os
import sys
sys.path.append(os.getcwd())
from nas_lib.trainer.trainer import NASBenchTrainer
from nas_lib.data import data
import tensorflow as tf
from nas_lib.utils.comm import set_random_seed, setup_logger
import time
import pickle
from configs import darts_converted_data_path


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)


def predictor_unsupervised(args, predictor_type, all_data, train_epochs=None, logger=None):
    trainer = NASBenchTrainer(args, predictor_type, len(all_data), train_epochs=train_epochs, logger=logger)
    start = time.time()
    trainer.fit_unsupervised(all_data)
    duration = time.time() - start
    logger.info(f'Self-supervised training time cost is {duration}.')


def main(args):
    file_name = 'log_%s_%d' % ('gpus', args.gpu)
    logger = setup_logger(file_name, args.save_dir, args.gpu, log_level='DEBUG',
                          filename='%s.txt' % file_name)
    logger.info(args)
    if args.search_space == 'darts':
        with open(args.darts_file_path, 'rb') as f:
            if args.darts_training_nums:
                all_data = pickle.load(f)[:args.darts_training_nums]
            else:
                all_data = pickle.load(f)
    else:
        nasbench_datas = data.build_datasets(args)
        all_data = data.dataset_all(args, nasbench_datas)
    for predictor in args.predictor_list:
        logger.info(f'==================  predictor type: {predictor}  ======================')
        predictor_unsupervised(args, predictor, all_data, train_epochs=args.epochs, logger=logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predictor comparison parameters!')
    #  unsupervised_ged: SS_RL
    parser.add_argument('--predictor_list', type=list, default=['SS_RL'],
                        help='The analysis architecture dataset type!')
    parser.add_argument('--seed', type=int, default=3344, help='The seed value.')
    parser.add_argument('--search_space', type=str, default='nasbench_101',
                        choices=['nasbench_101', 'nasbench_201', 'darts'],
                        help='The search space.')
    parser.add_argument('--epochs', type=int, default=300, help='The architecture training epochs.')
    parser.add_argument('--dataname', type=str, default='cifar10-valid',
                        choices=['cifar10-valid', 'cifar10', 'cifar100', 'ImageNet16-120'],
                        help='The evaluation of dataset of NASBench-201.')
    parser.add_argument('--darts_file_path', type=str,
                        default=darts_converted_data_path,
                        help='The evaluation of dataset of NASBench-201.')
    parser.add_argument('--darts_training_nums', type=int,
                        default=None,
                        help='The evaluation of dataset of NASBench-201.')
    parser.add_argument('--gpu', type=int, default=0, help='Choose which gpu to train the neural network.')
    parser.add_argument('--save_dir', type=str,
                        default='/home/albert_wei/Disk_A/train_output_2021/SS_RL_NASBENCH_101_300_5/',
                        help='The pre-trained unsupervised model save path.')
    parser.add_argument('--save_model', type=bool,
                        default=True,
                        help='The model save flag.')
    parser.add_argument('--full_train', type=bool,
                        default=False,
                        help='The model save flag.')
    parser.add_argument('--add_corresponding', type=bool,
                        default=False,
                        help='The model save flag.')
    parser.add_argument('--ged_type', type=str,
                        default='normalized', choices=['normalized', 'wo_normalized'],
                        help='The model save flag.')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help='How many architectures are used to train the unsupervised vae.')
    args = parser.parse_args()
    if args.search_space == 'nasbench_101':
        args.seq_len = 120
    elif args.search_space == 'nasbench_201':
        args.seq_len = 96
    else:
        raise NotImplementedError('This search space does not support at present!')

    set_random_seed(args.seed)

    main(args)
