import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"

# model_lists_nasbench = ['Random', 'EA', 'BANANAS', 'BANANAS_F', 'NPENAS-BO', 'NPENAS-NP']
# model_masks_nasbench = [True, True, True, True, True, True]


cnames = {
# 'aliceblue':            '#F0F8FF',
# 'antiquewhite':         '#FAEBD7',
'aqua':                 '#00FFFF',
'aquamarine':           '#7FFFD4',
'azure':                '#F0FFFF',
'beige':                '#F5F5DC',
'bisque':               '#FFE4C4',
'black':                '#000000',
'blanchedalmond':       '#FFEBCD',
'blue':                 '#0000FF',
'blueviolet':           '#8A2BE2',
'brown':                '#A52A2A',
'burlywood':            '#DEB887',
'cadetblue':            '#5F9EA0',
'chartreuse':           '#7FFF00',
'chocolate':            '#D2691E',
'coral':                '#FF7F50',
'cornflowerblue':       '#6495ED',
'cornsilk':             '#FFF8DC',
'crimson':              '#DC143C',
'cyan':                 '#00FFFF',
'darkblue':             '#00008B',
'darkcyan':             '#008B8B',
'darkgoldenrod':        '#B8860B',
'darkgray':             '#A9A9A9',
'darkgreen':            '#006400',
'darkkhaki':            '#BDB76B',
'darkmagenta':          '#8B008B',
'darkolivegreen':       '#556B2F',
'darkorange':           '#FF8C00',
'darkorchid':           '#9932CC',
'darkred':              '#8B0000',
'darksalmon':           '#E9967A',
'darkseagreen':         '#8FBC8F',
'darkslateblue':        '#483D8B',
'darkslategray':        '#2F4F4F',
'darkturquoise':        '#00CED1',
'darkviolet':           '#9400D3',
'deeppink':             '#FF1493',
'deepskyblue':          '#00BFFF',
'dimgray':              '#696969',
'dodgerblue':           '#1E90FF',
'firebrick':            '#B22222',
'floralwhite':          '#FFFAF0',
'forestgreen':          '#228B22',
'fuchsia':              '#FF00FF',
'gainsboro':            '#DCDCDC',
'ghostwhite':           '#F8F8FF',
'gold':                 '#FFD700',
'goldenrod':            '#DAA520',
'gray':                 '#808080',
'green':                '#008000',
'greenyellow':          '#ADFF2F',
'honeydew':             '#F0FFF0',
'hotpink':              '#FF69B4',
'indianred':            '#CD5C5C',
'indigo':               '#4B0082',
'ivory':                '#FFFFF0',
'khaki':                '#F0E68C',
'lavender':             '#E6E6FA',
'lavenderblush':        '#FFF0F5',
'lawngreen':            '#7CFC00',
'lemonchiffon':         '#FFFACD',
'lightblue':            '#ADD8E6',
'lightcoral':           '#F08080',
'lightcyan':            '#E0FFFF',
'lightgoldenrodyellow': '#FAFAD2',
'lightgreen':           '#90EE90',
'lightgray':            '#D3D3D3',
'lightpink':            '#FFB6C1',
'lightsalmon':          '#FFA07A',
'lightseagreen':        '#20B2AA',
'lightskyblue':         '#87CEFA',
'lightslategray':       '#778899',
'lightsteelblue':       '#B0C4DE',
'lightyellow':          '#FFFFE0',
'lime':                 '#00FF00',
'limegreen':            '#32CD32',
'linen':                '#FAF0E6',
'magenta':              '#FF00FF',
'maroon':               '#800000',
'mediumaquamarine':     '#66CDAA',
'mediumblue':           '#0000CD',
'mediumorchid':         '#BA55D3',
'mediumpurple':         '#9370DB',
'mediumseagreen':       '#3CB371',
'mediumslateblue':      '#7B68EE',
'mediumspringgreen':    '#00FA9A',
'mediumturquoise':      '#48D1CC',
'mediumvioletred':      '#C71585',
'midnightblue':         '#191970',
'mintcream':            '#F5FFFA',
'mistyrose':            '#FFE4E1',
'moccasin':             '#FFE4B5',
'navajowhite':          '#FFDEAD',
'navy':                 '#000080',
'oldlace':              '#FDF5E6',
'olive':                '#808000',
'olivedrab':            '#6B8E23',
'orange':               '#FFA500',
'orangered':            '#FF4500',
'orchid':               '#DA70D6',
'palegoldenrod':        '#EEE8AA',
'palegreen':            '#98FB98',
'paleturquoise':        '#AFEEEE',
'palevioletred':        '#DB7093',
'papayawhip':           '#FFEFD5',
'peachpuff':            '#FFDAB9',
'peru':                 '#CD853F',
'pink':                 '#FFC0CB',
'plum':                 '#DDA0DD',
'powderblue':           '#B0E0E6',
'purple':               '#800080',
'red':                  '#FF0000',
'rosybrown':            '#BC8F8F',
'royalblue':            '#4169E1',
'saddlebrown':          '#8B4513',
'salmon':               '#FA8072',
'sandybrown':           '#FAA460',
'seagreen':             '#2E8B57',
'seashell':             '#FFF5EE',
'sienna':               '#A0522D',
'silver':               '#C0C0C0',
'skyblue':              '#87CEEB',
'slateblue':            '#6A5ACD',
'slategray':            '#708090',
'snow':                 '#FFFAFA',
'springgreen':          '#00FF7F',
'steelblue':            '#4682B4',
'tan':                  '#D2B48C',
'teal':                 '#008080',
'thistle':              '#D8BFD8',
'tomato':               '#FF6347',
'turquoise':            '#40E0D0',
'violet':               '#EE82EE',
'wheat':                '#F5DEB3',
'white':                '#FFFFFF',
'whitesmoke':           '#F5F5F5',
'yellow':               '#FFFF00',
'yellowgreen':          '#9ACD32'}

c_names_list = list(cnames.keys())

color_names = ['dodgerblue', 'orange', 'green', 'firebrick', 'darkviolet', 'brown', 'pink', 'gray', 'gold',
               'skyblue', 'darkviolet', 'chocolate']

color_names_dict = {
    0: '#1f77b4',   # 31, 119, 180
    1: '#FF7F0E',   # 255, 127, 14
    2: '#2CA02C',   # 44, 160, 44
    3: '#D62728',   # 214, 39, 40
    4: '#9467BD',   # 148, 103, 189
    5: '#8c564b',   # 140, 86, 75
    6: '#E377C2',   # 227, 119, 194
    7: '#7F7F7F',   # 127, 127, 127
    8: '#BCBD22',   # 188, 189, 34
    9: '#17BECF',   # 23, 190, 207
    10: '#D76819',  # 215, 104, 25
}


def convert2np(root_path, model_lists, end=None):
    total_dicts = dict()
    for m in model_lists:
        total_dicts[m] = []
    files = os.listdir(root_path)
    if end:
        files = list(files)[:end]
    else:
        files = list(files)
    for f in files:
        if 'log' in f:
            continue
        file_path = os.path.join(root_path, f)
        nested_dicts = dict()
        for m in model_lists:
            nested_dicts[m] = []
        with open(file_path, 'rb') as nf:
            try:
                algorithm_params, metann_params, results, walltimes = pickle.load(nf)
                print('######')
            except Exception as e:
                print(e)
                print(file_path)
            for i in range(len(results[0])):
                for idx, m in enumerate(model_lists):
                    nested_dicts[m].append(results[idx][i][1])
            for m in model_lists:
                total_dicts[m].append(nested_dicts[m])
    results_np = {m: np.array(total_dicts[m]) for m in model_lists}
    return results_np


def getmean(results_np, model_lists, category='mean'):
    if category == 'mean':
        results_mean = {m: np.mean(results_np[m], axis=0) for m in model_lists}
    elif category == 'medium':
        results_mean = {m: np.median(results_np[m], axis=0) for m in model_lists}
    elif category == 'percentile':
        results_mean = {m: np.percentile(results_np[m], 50, axis=0) for m in model_lists}
    else:
        raise ValueError('this type operation is not supported!')
    return results_mean


def getstd(results_np, model_lists):
    result_std = {m: np.std(results_np[m], axis=0) for m in model_lists}
    return result_std


def get_quantile(results_np, model_lists, divider=30):
    results_quantile = {m: np.percentile(results_np[m], divider, axis=0) for m in model_lists}
    return results_quantile


def get_bounder(total_mean, quantile_30, quantile_70, model_lists, absolute=False):
    bound_dict = dict()
    for m in model_lists:
        bound_dict[m] = np.stack([(total_mean[m]-quantile_30[m]),
                                  (quantile_70[m]-total_mean[m])], axis=0)
    return bound_dict


def draw_plot_nasbench_101(root_path, model_lists, model_masks, draw_type='ERRORBAR', verbose=1):
    # draw_type  ERRORBAR, MEANERROR
    np_datas_dict = convert2np(root_path, model_lists=model_lists, end=None)
    # EA_reuslt = np_datas_dict['EA']
    # print(EA_reuslt.shape)
    # print(np.max(EA_reuslt, axis=0))
    # print(np.min(EA_reuslt, axis=0))
    np_mean_dict = getmean(np_datas_dict, model_lists=model_lists)
    np_std_dict = getstd(np_datas_dict, model_lists=model_lists)
    np_quantile_30 = get_quantile(np_datas_dict, model_lists=model_lists, divider=30)
    np_quantile_70 = get_quantile(np_datas_dict, model_lists=model_lists, divider=70)

    if verbose:
        for k, v in np_mean_dict.items():
            print(k)
            print('30 quantile value')
            print(np_quantile_30[k])
            print('mean')
            print(v)
            print('70 quantile value')
            print(np_quantile_70[k])
            print('std')
            print(np_std_dict[k])
            print('###############')
    np_bounds = get_bounder(np_mean_dict, np_quantile_30, np_quantile_70, model_lists=model_lists, absolute=True)
    # get data mean
    idx = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150])
    fig, ax = plt.subplots(1)
    fig.set_size_inches(6, 5)
    upperlimits = [True] * 15
    lowerlimits = [True] * 15
    if draw_type == 'ERRORBAR':
        for j, m in enumerate(model_lists):
            if model_masks[j]:
                # plt.errorbar(idx, np_mean_dict[m], yerr=np_bounds[m], uplims=upperlimits, lolims=lowerlimits,
                #              label=m, capthick=2)
                plt.errorbar(idx, np_mean_dict[m], yerr=np_bounds[m], label=m, capsize=3, capthick=2, color=color_names_dict[j])
    elif draw_type == 'MEANERROR':
        for j, m in enumerate(model_lists):
            if model_masks[j]:
                plt.plot(idx, np_mean_dict[m], label=m, marker='s', linewidth=1, ms=3)     # fmt='o',
    # ax.set_yticks(np.arange(92.5, 94.4, 0.2))
    ax.set_yticks(np.arange(5.8, 7.4, 0.2))
    # ax.grid(True)
    fig.set_dpi(300.0)
    ax.set_xlabel('number of samples', fontsize=13)
    ax.set_ylabel('testing error of best neural net', fontsize=13)
    plt.legend(loc='upper right', fontsize=12)
    # plt.grid(b=True, which='major', color='#666699', linestyle='--')
    plt.show()


def draw_plot_nasbench_201(root_path, model_lists, model_masks, draw_type='ERRORBAR', verbose=1, args=None):
    # draw_type  ERRORBAR, MEANERROR
    np_datas_dict = convert2np(root_path, end=None, model_lists=model_lists)
    np_std_dict = getstd(np_datas_dict, model_lists=model_lists)
    np_mean_dict = getmean(np_datas_dict, model_lists=model_lists)

    np_quantile_30 = get_quantile(np_datas_dict, model_lists=model_lists, divider=30)
    np_quantile_70 = get_quantile(np_datas_dict, model_lists=model_lists, divider=70)

    if verbose:
        for k, v in np_mean_dict.items():
            print(k)
            print('30 quantile value')
            print(np_quantile_30[k])
            print('mean')
            print(v)
            print('70 quantile value')
            print(np_quantile_70[k])
            print('std')
            print(np_std_dict[k])
            print('###############')
    np_bounds = get_bounder(np_mean_dict, np_quantile_30, np_quantile_70, model_lists=model_lists, absolute=True)
    # get data mean
    idx = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    fig, ax = plt.subplots(1)
    fig.set_size_inches(6, 5)

    upperlimits = [True] * 10
    lowerlimits = [True] * 10
    if draw_type == 'ERRORBAR':
        for j, m in enumerate(model_lists):
            if model_masks[j]:
                # plt.errorbar(idx, np_mean_dict[m], yerr=np_bounds[m], uplims=upperlimits, lolims=lowerlimits,
                #              label=m, capthick=2)
                plt.errorbar(idx, np_mean_dict[m], yerr=np_bounds[m], label=m, capsize=6, capthick=2, color=color_names_dict[j])
    elif draw_type == 'MEANERROR':
        for j, m in enumerate(model_lists):
            if model_masks[j]:
                plt.plot(idx, np_mean_dict[m], label=m, marker='s', linewidth=1, ms=3)     # fmt='o',
    if args.dataname == 'cifar10-valid':
        ax.set_yticks(np.arange(8.8, 10.9, 0.2))
    elif args.dataname == 'cifar100':
        ax.set_yticks(np.arange(26.5, 32.5, 0.5))
    elif args.dataname == 'ImageNet16-120':
        ax.set_yticks(np.arange(53.2, 58.2, 0.5))
    else:
        raise NotImplementedError()
    # ax.grid(True)
    fig.set_dpi(300.0)
    ax.set_xlabel('number of samples')
    ax.set_ylabel('testing error of best neural net')
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.5)
    plt.legend(loc='upper right')
    # plt.grid(b=True, which='major', color='#666699', linestyle='--')
    plt.show()


def draw_plot_nasbench_201_merge(root_path, root_path_2, model_lists, model_lists2,
                                 model_masks, model_masks2, draw_type='ERRORBAR', verbose=1, args=None):
    # draw_type  ERRORBAR, MEANERROR
    np_datas_dict = convert2np(root_path, end=None, model_lists=model_lists)
    np_std_dict = getstd(np_datas_dict, model_lists=model_lists)
    np_mean_dict = getmean(np_datas_dict, model_lists=model_lists)

    np_datas_dict2 = convert2np(root_path_2, end=None, model_lists=model_lists2)
    np_std_dict2 = getstd(np_datas_dict2, model_lists=model_lists2)
    np_mean_dict2 = getmean(np_datas_dict2, model_lists=model_lists2)

    for k, v in np_datas_dict2.items():
        np_datas_dict[k] = v

    for k, v in np_std_dict2.items():
        np_std_dict[k] = v

    for k, v in np_mean_dict2.items():
        np_mean_dict[k] = v
    # for k, v in np_d1atas_dict.items():
    #     data = v[:, -1]
    #     print(k)
    #     print(sorted(data.tolist()))
    #     print('######'*30)

    np_quantile_30 = get_quantile(np_datas_dict, model_lists=model_lists, divider=30)
    np_quantile_70 = get_quantile(np_datas_dict, model_lists=model_lists, divider=70)

    if verbose:
        for k, v in np_mean_dict.items():
            print(k)
            print('30 quantile value')
            print(np_quantile_30[k])
            print('mean')
            print(v)
            print('70 quantile value')
            print(np_quantile_70[k])
            print('std')
            print(np_std_dict[k])
            print('###############')
    np_bounds = get_bounder(np_mean_dict, np_quantile_30, np_quantile_70, model_lists=model_lists, absolute=True)
    # get data mean
    idx = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    fig, ax = plt.subplots(1)
    fig.set_size_inches(6, 5)
    # fig.set_size_inches(6, 9)

    upperlimits = [True] * 10
    lowerlimits = [True] * 10
    if draw_type == 'ERRORBAR':
        for j, m in enumerate(model_lists):
            if model_masks[j]:
                # plt.errorbar(idx, np_mean_dict[m], yerr=np_bounds[m], uplims=upperlimits, lolims=lowerlimits,
                #              label=m, capthick=2)
                plt.errorbar(idx, np_mean_dict[m], yerr=np_bounds[m], label=m, capsize=6, capthick=2, color=color_names_dict[j])
    elif draw_type == 'MEANERROR':
        for j, m in enumerate(model_lists):
            if model_masks[j]:
                plt.plot(idx, np_mean_dict[m], label=m, marker='s', linewidth=1, ms=3)     # fmt='o',
    if args.dataname == 'cifar10-valid':
        ax.set_yticks(np.arange(8.8, 10.9, 0.2))
    elif args.dataname == 'cifar100':
        ax.set_yticks(np.arange(26.5, 31.5, 0.5))
    elif args.dataname == 'ImageNet16-120':
        ax.set_yticks(np.arange(53.2, 58.2, 0.5))
    else:
        raise NotImplementedError()
    # ax.grid(True)
    fig.set_dpi(300.0)
    ax.set_xlabel('number of samples')
    ax.set_ylabel('testing error of best neural net')
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.5)
    # plt.legend(loc='upper right')
    # plt.grid(b=True, which='major', color='#666699', linestyle='--')
    plt.show()


def draw_plot_nasbench_101_merge(root_path, root_path_2, model_lists, model_lists2,
                                 model_masks, model_masks2, draw_type='ERRORBAR', verbose=1):
    # draw_type  ERRORBAR, MEANERROR
    np_datas_dict = convert2np(root_path, model_lists=model_lists, end=None)
    # EA_reuslt = np_datas_dict['EA']
    # print(EA_reuslt.shape)
    # print(np.max(EA_reuslt, axis=0))
    # print(np.min(EA_reuslt, axis=0))
    np_mean_dict = getmean(np_datas_dict, model_lists=model_lists)
    np_std_dict = getstd(np_datas_dict, model_lists=model_lists)

    np_datas_dict2 = convert2np(root_path_2, end=None, model_lists=model_lists2)
    np_std_dict2 = getstd(np_datas_dict2, model_lists=model_lists2)
    np_mean_dict2 = getmean(np_datas_dict2, model_lists=model_lists2)

    for k, v in np_datas_dict2.items():
        np_datas_dict[k] = v

    for k, v in np_std_dict2.items():
        np_std_dict[k] = v

    for k, v in np_mean_dict2.items():
        np_mean_dict[k] = v


    np_quantile_30 = get_quantile(np_datas_dict, model_lists=model_lists, divider=30)
    np_quantile_70 = get_quantile(np_datas_dict, model_lists=model_lists, divider=70)

    if verbose:
        for k, v in np_mean_dict.items():
            print(k)
            print('30 quantile value')
            print(np_quantile_30[k])
            print('mean')
            print(v)
            print('70 quantile value')
            print(np_quantile_70[k])
            print('std')
            print(np_std_dict[k])
            print('###############')
    np_bounds = get_bounder(np_mean_dict, np_quantile_30, np_quantile_70, model_lists=model_lists, absolute=True)
    # get data mean
    idx = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150])
    fig, ax = plt.subplots(1)
    fig.set_size_inches(6, 5)
    upperlimits = [True] * 15
    lowerlimits = [True] * 15
    if draw_type == 'ERRORBAR':
        for j, m in enumerate(model_lists):
            if model_masks[j]:
                # plt.errorbar(idx, np_mean_dict[m], yerr=np_bounds[m], uplims=upperlimits, lolims=lowerlimits,
                #              label=m, capthick=2)
                plt.errorbar(idx, np_mean_dict[m], yerr=np_bounds[m], label=m, capsize=3, capthick=2, color=color_names_dict[j])
    elif draw_type == 'MEANERROR':
        for j, m in enumerate(model_lists):
            if model_masks[j]:
                plt.plot(idx, np_mean_dict[m], label=m, marker='s', linewidth=1, ms=3)     # fmt='o',
    # ax.set_yticks(np.arange(92.5, 94.4, 0.2))
    ax.set_yticks(np.arange(5.8, 9.4, 0.2))
    # ax.grid(True)
    fig.set_dpi(300.0)
    ax.set_xlabel('number of samples', fontsize=13)
    ax.set_ylabel('testing error of best neural net', fontsize=13)
    plt.legend(loc='upper right', fontsize=12)
    # plt.grid(b=True, which='major', color='#666699', linestyle='--')
    plt.show()