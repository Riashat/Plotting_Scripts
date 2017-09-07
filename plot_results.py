import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
from itertools import cycle

from numpy import genfromtxt
from numpy.random import choice

plt.rcParams['text.usetex'] = True


def multiple_plot(average_vals_list, std_dev_list, traj_list, other_labels, env_name, smoothing_window=5, no_show=False, ignore_std=False, limit=None, extra_lines=None):
    #fig = plt.figure(figsize=(16, 8))
    fig = plt.figure(figsize=(15, 10))
    colors = ["#1f77b4", "#ff7f0e", "#d62728", "#9467bd", "#2ca02c", "#8c564b", "#e377c2", "#bcbd22", "#17becf"]
    #colors = ["#332288", "#88CCEE", "#999933", "#DDCC77", "#CC6677", "#882255", "#AA4499"]
    #colors = ["#332288", "#88CCEE", "#44AA99", "#117733", "#999933", "#DDCC77", "#CC6677", "#882255", "#AA4499"]
    #colors = ["#332288", "#117733", "#999933", "#CC6677", "#882255", "#AA4499"]
    #colors = ["#96595A", "#DA897C", "#E4E4B2", "#B2E4CF", "#0D6A82"]
    # colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#a6cee3"]
    #colors = list(reversed(colors))
    #colors = ["k", "red", "blue", "green", "magenta", "cyan", "brown", "purple"]
    color_index = 0
    ax = plt.subplot() # Defines ax variable by creating an empty plot
    offset = 1

    # Set the tick labels font
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(22)
    if traj_list is None:
        traj_list = [None]*len(average_vals_list)

    index = 0 
    for average_vals, std_dev, label, trajs in zip(average_vals_list, std_dev_list, other_labels[:len(average_vals_list)], traj_list):
        index += 1
        rewards_smoothed_1 = pd.Series(average_vals).rolling(smoothing_window, min_periods=smoothing_window).mean()[:limit]
        if limit is None:
            limit = len(rewards_smoothed_1)
        rewards_smoothed_1 = rewards_smoothed_1[:limit]
        std_dev = std_dev[:limit]
        if trajs is None:
            trajs = list(range(len(rewards_smoothed_1)))
        else:
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

            ax.xaxis.get_offset_text().set_fontsize(20)

        fill_color = colors[color_index]#choice(colors, 1)
        color_index += 1
        cum_rwd_1, = plt.plot(trajs, rewards_smoothed_1, label=label, color=fill_color)
        offset += 3
        if not ignore_std:
            #plt.errorbar(trajs[::25 + offset], rewards_smoothed_1[::25 + offset], yerr=std_dev[::25 + offset], linestyle='None', color=fill_color, capsize=5)
            plt.fill_between(trajs, rewards_smoothed_1 + std_dev,   rewards_smoothed_1 - std_dev, alpha=0.3, edgecolor=fill_color, facecolor=fill_color)

    if extra_lines:
        for lin in extra_lines:
            plt.plot(trajs, np.repeat(lin, len(rewards_smoothed_1)), linestyle='-.', color = colors[color_index], linewidth=2.5, label=other_labels[index])
            color_index += 1
            index += 1

    axis_font = {'fontname':'Arial', 'size':'28'}
    #plt.legend(loc='upper left', prop={'size' : 16})
    plt.legend(loc='lower right', prop={'size' : 16})
    plt.xlabel("Iterations", **axis_font)
    if traj_list:
        plt.xlabel("Timesteps", **axis_font)
    else:
        plt.xlabel("Iterations", **axis_font)
    plt.ylabel("Average Return", **axis_font)
    plt.title("%s"% env_name, **axis_font)

    if no_show:
        fig.savefig('%s.png' % env_name, dpi=fig.dpi)
    else:
        plt.show()

    return fig

# def multipe_plot(stats1, stats2, smoothing_window=50, noshow=False):
#
#     fig = plt.figure(figsize=(30, 20))
#     rewards_smoothed_1 = pd.Series(stats1).rolling(smoothing_window, min_periods=smoothing_window).mean()
#
#     rewards_smoothed_2 = pd.Series(stats2).rolling(smoothing_window, min_periods=smoothing_window).mean()
#
#     cum_rwd_1, = plt.plot(eps, rewards_smoothed_1, label="DDPG")
#     plt.fill_between( eps, rewards_smoothed_1 + ddpg_walker_std_return,   rewards_smoothed_1 - ddpg_walker_std_return, alpha=0.3, edgecolor='blue', facecolor='blue')
#
#     cum_rwd_2, = plt.plot(eps2, rewards_smoothed_2, label="Unified DDPG")
#     plt.fill_between( eps2, rewards_smoothed_2 + unified_ddpg_walker_std_return,   rewards_smoothed_2 - unified_ddpg_walker_std_return, alpha=0.3, edgecolor='blue', facecolor='red')
#
#     plt.legend(handles=[cum_rwd_1, cum_rwd_2])
#     plt.xlabel("Epsiode")
#     plt.ylabel("Average Return")
#     plt.title("Walker Environment")
#
#     plt.show()
#
#     return fig






import argparse
parser = argparse.ArgumentParser()
parser.add_argument("paths_to_progress_csvs", nargs="+", help="All the csvs")
parser.add_argument("env_name")
parser.add_argument("--save", action="store_true")
parser.add_argument("--ignore_std", action="store_true")
parser.add_argument('--labels', nargs='+', help='List of labels to go along with the paths', required=False)
parser.add_argument('--smoothing_window', default=5, type=int)
parser.add_argument('--limit', default=None, type=int)
parser.add_argument('--extra_lines', nargs="+", type=float)

args = parser.parse_args()

avg_rets = []
std_dev_rets = []
trajs = []

for o in args.paths_to_progress_csvs:
    data = pd.read_csv(o)
    avg_ret = np.array(data["TrueAverageReturn"])
    std_dev_ret = np.array(data["TrueStdReturn"])
    if "total/steps" in data:
        trajs.append(np.array(data["total/steps"]))
    elif "TimestepsSoFar" in data:
        trajs.append(np.array(data["TimestepsSoFar"]))
    else:
        trajs=None
        #trajs.append(np.cumsum(np.array([25]*len(data["TrueAverageReturn"]))))
    avg_rets.append(avg_ret)
    std_dev_rets.append(std_dev_ret)

multiple_plot(avg_rets, std_dev_rets, trajs, args.labels, args.env_name, smoothing_window=args.smoothing_window, no_show=args.save, ignore_std=args.ignore_std, limit=args.limit, extra_lines=args.extra_lines)
