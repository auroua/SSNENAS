import os
import sys
import argparse
sys.path.append(os.getcwd())
from nas_lib.visualization.visualize_close_domain import draw_plot_nasbench_101, draw_plot_nasbench_201


model_lists_nasbench = ['RA', 'REA', 'BANANAS-PE', 'BANANAS-AE', 'BANANAS-PEAE', 'NPENAS-NP', 'NPENAS-NP-FIXED',
                        'NPENAS-SSRL', 'NPENAS-SSRL-FIXED', 'NPENAS-SSCCL', 'NPENAS-SSCCL-FIXED']
model_masks_nasbench = [True, True, True, True, True, True, True, True, True, True, True]
#
#
# # NASBench-101
# model_lists_nasbench = ['NPENAS-SSRL-20', 'NPENAS-SSRL-50', 'NPENAS-SSRL-80', 'NPENAS-SSRL-110', 'NPENAS-SSRL-150',
#                         'NPENAS-SSCCL-20', 'NPENAS-SSCCL-50', 'NPENAS-SSCCL-80', 'NPENAS-SSCCL-110', 'NPENAS-SSCCL-150']
# model_masks_nasbench = [True, True, True, True, True, True, True, True, True, True, True]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args for visualize darts architecture')
    parser.add_argument('--search_space', type=str, default='nasbench_101',
                        choices=['nasbench_101', 'nasbench_201'],
                        help='The algorithm output folder')
    parser.add_argument('--save_dir', type=str,
                        default='/home/aurora/Desktop/ssnenas_revise_results/npenas_results/npenas_fixed_num_nasbench_101/',
                        # default='/home/aurora/Desktop/ssnenas_revise_results/npenas_results/npenas_fixed_num_nasbench_101_batch_size_compare/',
                        # nasbench-201 fixed num
                        # default='/home/aurora/Desktop/ssnenas_revise_results/npenas_results/npenas_fixed_num_nasbench_201_cifar100/',
                        # default='/home/aurora/Desktop/ssnenas_revise_results/npenas_results/npenas_fixed_num_nasbench_201_imagenet/',
                        # default='/home/aurora/Desktop/ssnenas_revise_results/npenas_results/npenas_fixed_num_nasbench_201_cifar10_valid/',
                        help='The algorithm output folder')
    parser.add_argument('--draw_type', type=str, default='ERRORBAR', choices=['ERRORBAR', 'MEANERROR'],
                        help='Draw result with or without errorbar.')
    parser.add_argument('--dataname', type=str, default='ImageNet16-120',
                        choices=['cifar10-valid', 'cifar10', 'cifar100', 'ImageNet16-120'],
                        help='The evaluation of dataset of NASBench-201.')
    parser.add_argument('--show_all', type=str, default='1', help='Weather to show all results.')
    args = parser.parse_args()
    if args.search_space == 'nasbench_101':
        draw_plot_nasbench_101(args.save_dir, draw_type=args.draw_type, model_lists=model_lists_nasbench,
                               model_masks=model_masks_nasbench)
    elif args.search_space == 'nasbench_201':
        draw_plot_nasbench_201(args.save_dir, draw_type=args.draw_type, model_lists=model_lists_nasbench,
                               model_masks=model_masks_nasbench, args=args)
    else:
        raise ValueError('This search space does not support!')