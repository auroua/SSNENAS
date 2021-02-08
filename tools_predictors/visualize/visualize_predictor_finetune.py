import matplotlib.pyplot as plt
import pickle
import numpy as np
import argparse
from collections import defaultdict
import os


plt.rcParams["font.family"] = "Times New Roman"
marker_list = ['*', '^', '1', 's', '+', 'x', 'o', 'v', '2', 'D', 'd', 'p']


def parse_pkl_2d(s_dict, k_dict, coor_type='s', data_type='epoch', other_dim_val=(100,),
                 unsupervised_val_list=None, show_keys=None):
    data_dict = defaultdict(dict)
    print(f'========== Experiment iteration is {len(s_dict[list(s_dict.keys())[0]])}. ============')
    if coor_type == 's':
        d = s_dict
    else:
        d = k_dict
    for k, v in d.items():
        key, budget, epoch = k.split('#')
        # print(key, budget, epoch)
        if key not in show_keys:
            continue
        if data_type == 'epoch':
            if int(budget) not in other_dim_val:
                continue
            if 'unsupervised' in key:
                if int(budget) not in unsupervised_val_list:
                    continue
            dict_key = key+'_'+str(budget)
        else:
            if int(epoch) not in other_dim_val:
                continue
            if 'unsupervised' in key:
                if int(epoch) not in unsupervised_val_list:
                    continue
            dict_key = key + '_' + str(epoch)
        if data_type == 'epoch':
            if 'x' not in data_dict[dict_key]:
                data_dict[dict_key]['x'] = [epoch]
            else:
                data_dict[dict_key]['x'].append(epoch)
        else:
            if 'x' not in data_dict[dict_key]:
                data_dict[dict_key]['x'] = [budget]
            else:
                data_dict[dict_key]['x'].append(budget)
        if 'v' not in data_dict[dict_key]:
            data_dict[dict_key]['v'] = [np.mean(np.array(d[k]))]
        else:
            data_dict[dict_key]['v'].append(np.mean(np.array(d[k])))
    return data_dict


def draw_2d_results(s_dict, k_dict, coor_type='s', data_type='epoch', other_dim_val=(100,), unsupervised_val_list=(100,),
                    show_keys=None, append_data=None, key_mapping=None):
    data_dict = parse_pkl_2d(s_dict, k_dict, coor_type=coor_type, data_type=data_type, other_dim_val=other_dim_val,
                             unsupervised_val_list=unsupervised_val_list, show_keys=show_keys)
    if key_mapping and append_data is not None and 'SS_RL' not in show_keys:
        for k_a, v_a in append_data.items():
            data_dict[k_a] = v_a
            show_keys.append('_'.join(k_a.split('_')[:-1]))
    fig, ax = plt.subplots()
    fig.set_size_inches(5, 4.4)
    if key_mapping and 'SS_RL' not in show_keys:
        keys = sorted(list(data_dict.keys()))
        suffiex = keys[0].split('_')[-1]
        keys = [k + '_' + suffiex for k in show_keys]
    else:
        keys = []
        keys_temp = sorted(list(data_dict.keys()))
        for k in keys_temp:
            if 'supervised' in k:
                keys.append(k)
        for k in keys_temp:
            if 'SS_RL' in k:
                keys.append(k)
        for k in keys_temp:
            if 'SS_CCL' in k:
                keys.append(k)
    for idx, k in enumerate(keys):
        print(k)
        print(data_dict[k])
        print('##################')
        if 'supervised' in k and 'unsupervised' not in k:
            label_k = 'SUPERVISED'
        elif 'SS_RL' in k:
            label_k = 'SS-RL'
        elif 'SS_CCL' in k and 'k' not in k:
            label_k = 'SS-CCL'
        elif 'SS_CCL' in k and 'k' in k:
            label_k = '_'.join(k.split('_')[:-1])
        else:
            raise NotImplementedError()
        ax.scatter(data_dict[k]['x'], data_dict[k]['v'], marker=marker_list[idx], label=label_k, s=100)
        ax.plot(data_dict[k]['x'], data_dict[k]['v'], linestyle='dashed', linewidth=2)
    if data_type == 'epoch':
        ax.set_xlabel('training epochs', loc='center', fontsize=14)
    else:
        ax.set_xlabel('search budget', loc='center', fontsize=14)
    if coor_type == 's':
        ylabel = 'Spearman correlation'
    elif coor_type == 'k':
        ylabel = 'Kendall tau correlation'
    else:
        raise ValueError('This coorlation type does not support at present!')
    ax.set_yticks(np.arange(0., 0.7, 0.1))

    ax.set_ylabel(ylabel, loc='center', fontsize=14)
    if data_type == 'epoch':
        title = f'search budget {other_dim_val[0]}'
    else:
        title = f'training epoch: {other_dim_val[0]}'
    plt.title(title, fontsize=14)
    fig.set_dpi(300.0)
    # upper left lower right
    if key_mapping:
        plt.legend(loc='lower right', fontsize=12)
    else:
        plt.legend(loc='lower right', fontsize=12)
    plt.show()


def merge_files(base_path):
    file_path = [os.path.join(base_path, p) for p in os.listdir(base_path) if not p.endswith('.txt')]

    merge_s_dict = defaultdict(list)
    merge_k_dict = defaultdict(list)

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
                        # default='/home/aurora/Desktop/ssnenas_revise_results/predictor_compare_results/predictor_finetune_comapre_nasbench_101/',
                        default='/home/aurora/Desktop/ssnenas_revise_results/predictor_compare_results/predictor_finetune_comapre_nasbench_201/',
                        help='The analysis architecture dataset type!')
    args = parser.parse_args()

    s_dict, k_dict = merge_files(args.result_path)

    for idx, i in enumerate([50, 100, 150, 200, 250, 300]):
        draw_2d_results(s_dict,
                        k_dict,
                        coor_type='k',
                        data_type='budget',
                        other_dim_val=[i],
                        show_keys=['supervised',
                                   'SS_RL',
                                   'SS_CCL',
                                   ],
                        unsupervised_val_list=[i],
                        append_data=None
                        )