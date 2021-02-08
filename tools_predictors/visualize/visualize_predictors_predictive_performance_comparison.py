import matplotlib.pyplot as plt
import pickle
import numpy as np
import argparse
from collections import defaultdict
import os
import math


plt.rcParams["font.family"] = "Times New Roman"
marker_list = ['*', '^', '1', '+', 'x', 'v', '2', 'D', 'd', 'p', 'o', 's']
# k: black m: purple b: blue g: green r: red
color_list = ['darkorange', 'g', 'r', 'b', 'brown', 'm', 'c', 'y',  'blueviolet', 'k','slategray',
              'olive', 'dodgerblue']


def parse_pkl_2d(s_dict, k_dict, coor_type='s', show_keys=None, search_budget=None):
    data_dict = defaultdict(dict)
    result_dict = defaultdict(list)
    print(f'========== Experiment iteration is {len(s_dict[list(s_dict.keys())[0]])}. ============')
    if coor_type == 's':
        d = s_dict
    else:
        d = k_dict
    for k, v in d.items():
        key, budget, epoch = k.split('#')
        if key not in show_keys:
            continue
        dict_key = key+'_'+str(budget)

        if 'v' not in data_dict[dict_key]:
            data_dict[dict_key]['v'] = [np.mean(np.array(d[k]))]
        else:
            data_dict[dict_key]['v'].append(np.mean(np.array(d[k])))

    for k in show_keys:
        for budget in search_budget:
            if math.isnan(data_dict[k+'_'+str(budget)]['v'][0]):
                val = {'v': [0]}
            else:
                val = data_dict[k+'_'+str(budget)]
            if k == 'supervised':
                result_dict['SUPERVISED'].append(val['v'][0])
            else:
                result_dict[k].append(val['v'][0])
    return result_dict


def draw_2d_results(s_dict, k_dict, coor_type='s', show_keys=None, search_budget=None, search_space='NASBench_101'):
    data_dict = parse_pkl_2d(s_dict, k_dict, coor_type=coor_type, show_keys=show_keys, search_budget=search_budget)
    fig, ax = plt.subplots()
    fig.set_size_inches(5, 4.4)
    keys = sorted(list(data_dict.keys()))
    for idx, k in enumerate(keys):

        if 'SS_RL' in k:
            label_k = 'SS-RL'
        elif 'SS_CCL' in k:
            label_k = 'SS-CCL'
        else:
            label_k = k
        marker = marker_list[idx]
        color = color_list[idx]
        for j, budget in enumerate(search_budget):
            ax.scatter(budget, data_dict[k][j], marker=marker, label=label_k, c=color, s=100)
        ax.plot(search_budget, data_dict[k], linestyle='dashed', linewidth=2, c=color)
        print(color, k)
    ax.set_xlabel('search budget', loc='center', fontsize=14)
    if coor_type == 's':
        ylabel = 'Spearman correlation'
    elif coor_type == 'k':
        ylabel = 'Kendall tau correlation'
    else:
        raise ValueError('This coorlation type does not support at present!')
    if search_space == '101':
        ax.set_yticks(np.arange(0., 0.65, 0.1))
    elif search_space == '201':
        ax.set_yticks(np.arange(0., 0.85, 0.1))
    ax.set_ylabel(ylabel, loc='center', fontsize=13)
    title = f'training epoch: {200}'
    plt.title(title, fontsize=13)
    fig.set_dpi(300.0)
    # upper left lower right
    # plt.legend(loc='best', fontsize=12)
    plt.show()


def merge_files(base_path):
    file_path = [os.path.join(base_path, p) for p in os.listdir(base_path) if not p.endswith('.txt')]

    merge_s_dict = defaultdict(list)
    merge_k_dict = defaultdict(list)
    merge_duration_dict = defaultdict(list)

    for fp in file_path:
        with open(fp, 'rb') as f:
            s1_dict = pickle.load(f)
            k1_dict = pickle.load(f)

            for k, v in s1_dict.items():
                merge_s_dict[k].extend(v)

            for k, v in k1_dict.items():
                merge_k_dict[k].extend(v)

    return merge_s_dict, merge_k_dict


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predictor comparison parameters!')
    parser.add_argument('--result_path', type=str,
                        default='/home/aurora/Desktop/ssnenas_revise_results/predictor_compare_results/predictors_performance_compare_seminas_np_nas_mlp/',
                        help='The analysis architecture dataset type!')
    parser.add_argument('--ss', type=str,
                        default='101',
                        help='The analysis architecture dataset type!')
    args = parser.parse_args()

    s_dict, k_dict = merge_files(args.result_path)

    search_budget = [20, 50, 100, 200]
    if args.ss == '101':
        keys = ['supervised',
                'SS_RL',
                'SS_CCL',
                'SemiNAS',
                'NP_NAS',
                'MLP',
                ]
    else:
        raise NotImplementedError()

    draw_2d_results(s_dict,
                    k_dict,
                    coor_type='k',
                    show_keys=keys,
                    search_budget=search_budget,
                    search_space=args.ss
                    )