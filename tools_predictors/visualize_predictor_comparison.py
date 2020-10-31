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
        keys = sorted(list(data_dict.keys()))
    for idx, k in enumerate(keys):
        print(k)
        print(data_dict[k])
        print('##################')

        temp_k = '_'.join(k.split('_')[:-1])
        if 'supervised' in k and 'unsupervised' not in k:
            label_k = 'SUPERVISED'
        elif 'SS_RL' in k :
            label_k = 'SS-RL'
        elif temp_k in key_mapping:
            label_k = key_mapping[temp_k]
        elif 'SS_CCL' in k:
            label_k = 'SS-CCL'
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
    ax.set_yticks(np.arange(0., 0.8, 0.1))

    ax.set_ylabel(ylabel, loc='center', fontsize=14)
    if data_type == 'epoch':
        title = f'search budget {other_dim_val[0]}'
    else:
        title = f'training epoch: {other_dim_val[0]}'
    plt.title(title, fontsize=14)
    fig.set_dpi(300.0)
    # upper left lower right
    if key_mapping:
        plt.legend(loc='best', fontsize=12)
    else:
        plt.legend(loc='best', fontsize=12)
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
                        default='/home/albert_wei/Disk_A/train_output_ssne_nas/test/',
                        help='The analysis architecture dataset type!')
    args = parser.parse_args()

    s_dict, k_dict = merge_files(args.result_path)

    data = [
        {'SS_CCL_14_50': {
            'x': ['20', '50', '100', '150', '200'],
            'v': [0.07460664515819564, 0.15781454759569685, 0.25986679734449974, 0.3093714979557737, 0.3441117847239889]
        }},
        {'SS_CCL_14_100': {
            'x': ['20', '50', '100', '150', '200'],
            'v': [0.09692171114354427, 0.18329940892143987, 0.2802203060653741, 0.334597785339594, 0.3869012883750581]
        }},
        {'SS_CCL_14_150': {
            'x': ['20', '50', '100', '150', '200'],
            'v': [0.1217463153848594, 0.17663019231398225, 0.29525138259855277, 0.35722640124935445, 0.4446601940670044]
        }},
        {'SS_CCL_14_200': {
            'x': ['20', '50', '100', '150', '200'],
            'v': [0.12506684111790972, 0.1965899347997259, 0.32421739781349157, 0.42525954011548367, 0.49081045901067843]
        }},
        {'SS_CCL_14_250': {
            'x': ['20', '50', '100', '150', '200'],
            'v': [0.11865825706409346, 0.16126832389395104, 0.3721133106409429, 0.4815361121479166, 0.503587459069718]
        }},
        {'SS_CCL_14_300': {
            'x': ['20', '50', '100', '150', '200'],
            'v': [0.08161209004336198, 0.1672639438447984, 0.4106900196825126, 0.4913961693545099, 0.5157615762547556]
        }}
    ]

    data_sp = [
        {'SS_CCL_14_50': {
            'x': ['20', '50', '100', '150', '200'],
            'v': [0.1100940099171932, 0.23226375731350207, 0.37962240026789884, 0.447937220185039, 0.4956499236406374]
        }},
        {'SS_CCL_14_100': {
            'x': ['20', '50', '100', '150', '200'],
            'v': [0.14264655354783934, 0.269128385298806, 0.40719463056594307, 0.4829580009100754, 0.5502833271873088]
        }},
        {'SS_CCL_14_150': {
            'x': ['20', '50', '100', '150', '200'],
            'v': [0.1790895869704452, 0.2605562642824533, 0.4281006081211549, 0.5101267993364345, 0.6216202516345787]
        }},
        {'SS_CCL_14_200': {
            'x': ['20', '50', '100', '150', '200'],
            'v': [0.18471540611993464, 0.2897086709222683, 0.46647269355811255, 0.5964526363409683, 0.675507231031883]
        }},
        {'SS_CCL_14_250': {
            'x': ['20', '50', '100', '150', '200'],
            'v': [0.17501769261822225, 0.23828721400029673, 0.5288987482885054, 0.6628148399954468, 0.6899871600412556]
        }},
        {'SS_CCL_14_300': {
            'x': ['20', '50', '100', '150', '200'],
            'v': [0.12068830240039648, 0.2471571783130687, 0.5781995349778861, 0.6753343905063633, 0.7034118913516171]
        }}
    ]

    key_mapping = {
        'SS_CCL_1': 'SS-CCL_10k',
        'SS_CCL_4': 'SS-CCL_40k',
        'SS_CCL_7': 'SS-CCL_70k',
        'SS_CCL_10': 'SS-CCL_100k',
        'SS_CCL_14': 'SS-CCL_140k'
    }

    # for idx, i in enumerate([50, 100, 150, 200, 250, 300]):
    for idx, i in enumerate([50]):
        draw_2d_results(s_dict,
                        k_dict,
                        coor_type='k',
                        data_type='budget',
                        other_dim_val=[i],
                        show_keys=['supervised',
                                   'SS_RL',
                                   'SS_CCL',
                                   ],
                        # show_keys=['supervised',
                        #            'SS_CCL_1',
                        #            'SS_CCL_4',
                        #            'SS_CCL_7',
                        #            'SS_CCL_10',
                        #            ],
                        unsupervised_val_list=[i],
                        append_data=data[idx],
                        key_mapping=key_mapping
                        )