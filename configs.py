# Copyright (c) Xidian University and Xi'an University of Posts & Telecommunications. All Rights Reserved

import os

nas_bench_101_base_path = '/home/albert_wei/Disk_A/dataset_train/nas_bench101/'
tf_records_path_108 = os.path.join(nas_bench_101_base_path, 'nasbench_only108.tfrecord')
nas_bench_101_converted_file_path = os.path.join(nas_bench_101_base_path, 'nasbench_archs.pkl')
nas_bench_101_all_data = os.path.join(nas_bench_101_base_path, 'all_data_new.pkl')

nas_bench_201_base_path = '/home/albert_wei/Disk_A/dataset_train/nas_bench_201/'
nas_bench_201_path = os.path.join(nas_bench_201_base_path, 'NAS-Bench-201-v1_1-096897.pth')

ss_rl_nasbench_101 = '/home/albert_wei/Disk_A/train_output_ssne_nas/results/pre_trained_models/ss_rl_nasbench_101/unsupervised_ged_epoch_299.pt'
ss_rl_nasbench_201 = '/home/albert_wei/Disk_A/train_output_ssne_nas/results/pre_trained_models/ss_rl_nasbench_201/unsupervised_ged_epoch_299.pt'

ss_ccl_nasbench_101 = '/home/albert_wei/Disk_A/train_output_ssne_nas/results/pre_trained_models/ss_ccl_nasbench_101_140000_1399800_wo_margin/checkpoint_0299.pth.tar'
ss_ccl_nasbench_201 = '/home/albert_wei/Disk_A/train_output_ssne_nas/results/pre_trained_models/ss_ccl_nasbench_201/checkpoint_0299.pth.tar'