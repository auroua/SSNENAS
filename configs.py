# Copyright (c) Xidian University and Xi'an University of Posts & Telecommunications. All Rights Reserved

import os

nas_bench_101_base_path = '/home/albert_wei/Disk_A/dataset_train/nas_bench101/'
tf_records_path_108 = os.path.join(nas_bench_101_base_path, 'nasbench_only108.tfrecord')
nas_bench_101_converted_file_path = os.path.join(nas_bench_101_base_path, 'nasbench_archs.pkl')
nas_bench_101_all_data = os.path.join(nas_bench_101_base_path, 'all_data_new.pkl')

nas_bench_201_base_path = '/home/albert_wei/Disk_A/dataset_train/nas_bench_201/'
nas_bench_201_path = os.path.join(nas_bench_201_base_path, 'NAS-Bench-201-v1_1-096897.pth')

darts_converted_data_path = '/home/albert_wei/Disk_A/dataset_train/nas_bench_301/gen_archs/data_info_part1.pkl'
darts_converted_with_label = '/home/albert_wei/Disk_A/dataset_train/nas_bench_301/convert_label_all_data_partial.pkl'

ss_rl_nasbench_101 = '/home/albert_wei/Disk_A/train_output_2021/SS_RL_NASBENCH_101_300_5/unsupervised_ss_rl_epoch_299.pt'
ss_rl_nasbench_201 = '/home/albert_wei/Disk_A/train_output_2021/ssnenas_pre_trained_models_revise/nasbench_201/SS_RL_NAS_BENCH_201_1000/unsupervised_ss_rl_epoch_999.pt'

ss_ccl_nasbench_101 = '/home/albert_wei/Disk_A/train_output_2021/ssnenas_pre_trained_models_revise/nasbench_101/ss_ccl_nasbench_101_140000_1399800_wo_margin/checkpoint_0282.pth.tar'
ss_ccl_nasbench_201 = '/home/albert_wei/Disk_A/train_output_2021/ssnenas_pre_trained_models_revise/nasbench_201/ss_ccl_nasbench_201/checkpoint_0299.pth.tar'
ss_ccl_darts = '/home/albert_wei/Disk_A/train_output_2021/ssnenas_pre_trained_models_revise/darts/SS_CCL_DARTS/checkpoint_0299.pth.tar'

cifar10_path = '/home/albert_wei/Disk_A/dataset_train/cifar10/'

ss_ccl_nasbench_10k = '/home/albert_wei/Disk_A/train_output_2021/ssnenas_pre_trained_models_revise/batch_size_compare/ccl_nasbench_101_10000_5000/checkpoint_0299.pth.tar'
ss_ccl_nasbench_40k = '/home/albert_wei/Disk_A/train_output_2021/ssnenas_pre_trained_models_revise/batch_size_compare/ccl_nasbench_101_40000_20000/checkpoint_0299.pth.tar'
ss_ccl_nasbench_70k = '/home/albert_wei/Disk_A/train_output_2021/ssnenas_pre_trained_models_revise/batch_size_compare/ccl_nasbench_101_70000_35000/checkpoint_0299.pth.tar'
ss_ccl_nasbench_100k = '/home/albert_wei/Disk_A/train_output_2021/ssnenas_pre_trained_models_revise/batch_size_compare/ccl_nasbench_101_100000_50000/checkpoint_0278.pth.tar'
ss_ccl_nasbench_140k = '/home/albert_wei/Disk_A/train_output_2021/ssnenas_pre_trained_models_revise/nasbench_101/ss_ccl_nasbench_101_140000_1399800_wo_margin/checkpoint_0282.pth.tar'

ss_rl_wo_ged_1000_nasbench_201 = '/home/aurora/data_disk_new/train_output_2021/ssnenas_pre_trained_models_revise/nasbench_201/ss_rl_wo_normalized_ged_1000_nasbench_201/unsupervised_ss_rl_epoch_999.pt'
