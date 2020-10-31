# Copyright (c) Xidian University and Xi'an University of Posts & Telecommunications. All Rights Reserved


import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"


def visual_correlation(v1, v2, xlabel, ylabel, title):
    fig, ax = plt.subplots()
    ax.scatter(v1, v2, label='', alpha=0.7, s=3, color='tab:green')
    # axis limits
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    ax_lim = np.array([np.min([plt.xlim()[0], plt.ylim()[0]]),
                       np.max([plt.xlim()[1], plt.ylim()[1]])])
    plt.xlim(ax_lim)
    plt.ylim(ax_lim)

    # 45-degree line
    ax.plot(ax_lim, ax_lim, 'k:')
    fig.set_dpi(300.0)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)
    # plt.legend(loc='best', fontsize=12)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    plt.show()


def visual_hist(x, num_bins=100, title=None):
    fig, ax = plt.subplots()
    # the histogram of the data
    ax.hist(x, num_bins, density=True)

    fig.set_dpi(300.0)
    ax.set_xlabel('Data', fontsize=12)
    ax.set_ylabel('Probability density', fontsize=12)
    ax.set_title(title, fontsize=12)

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()


def visual_plot(x, y_val, v_std, y_test=None, t_std=None):
    fig, ax = plt.subplots()
    # ax.errorbar(x, y_val, yerr=v_std)
    ax.plot(x, y_val)
    # ax.errorbar(x, y_test, yerr=t_std)
    plt.show()