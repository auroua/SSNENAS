from nas_lib.ccl.ccl_nas import main
import argparse
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)


parser = argparse.ArgumentParser(description='PyTorch Center Contrastive Learning Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='SS_CCL',
                    choices=['SS_CCL'],
                    help='The supported unsupervised model type!')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.005, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=444, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', default=False,
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--moco_dim_fc', default=32, type=int,
                    help='feature dimension (default: 64)')
parser.add_argument('--moco-dim', default=8, type=int,
                    help='feature dimension (default: 32)')

# options for moco v2
parser.add_argument('--cos', default=True, help='use cosine lr schedule')

# nas parameters
parser.add_argument('--search_space', default='nasbench_101',
                    choices=['nasbench_101', 'nasbench_201'],
                    help='The nasbench benchmark.')
# parser.add_argument('--input_dim', default=6, type=int,
#                     help='input feature dimension. nasbench_101: 6, nasbench_201: 8')
parser.add_argument('--save_dir',
                    default='/home/albert_wei/Disk_A/train_output_ssne_nas/test/',
                    help='The save dir of ckpt models.')

# nasbench-101: batch-size: 4000,  train_samples: 1000,  margin: 2, min_negative_size: 3500
# nasbench-201: batch-size: 5000,  train_samples: 1000,  margin: 2, min_negative_size: 4500
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('-b', '--batch-size', default=10000, type=int,
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--train_samples', default=5000, type=int,
                    help='How many samples used to train this model.')
parser.add_argument('--batch_step', default=1000, type=int,
                    help='How many samples used to train this model.')
parser.add_argument('--margin', default=0, type=int,
                    help='The margin between positive and negative pair.')
parser.add_argument('--min_negative_size', default=9900, type=int,
                    help='The minimum number of negative samples.')
parser.add_argument('--center_regularization', default=True, type=bool,
                    help='The minimum number of negative samples.')
parser.add_argument('--with_g_func', type=bool, default=False,
                    help='Using the g function after the backbone.')
parser.add_argument('--gpu_count', type=int, default=0,
                    help='Using the g function after the backbone.')
parser.add_argument('--gpu_base', type=int, default=0,
                    help='Using the g function after the backbone.')

if __name__ == '__main__':
    # For multiple gpu training, you should modify the following parameters: set gpu to None,
    # set multiprocessing-distributed to True

    args = parser.parse_args()

    if args.search_space == 'nasbench_101':
        args.input_dim = 6
    elif args.search_space == 'nasbench_201':
        args.input_dim = 8
    else:
        raise NotImplementedError('Does support this search space at present!')

    main(args)