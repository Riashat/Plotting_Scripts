import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
from numpy import genfromtxt
import pdb
from scipy import stats


eps = np.arange(1000)


#HalfCheetah Policy Activations
hs_leaky_relu = np.load('/Users/Riashat/Documents/PhD_Research/OpenAIBaselines/ReproducibilityML/Results/rllab_results/baselines_ddpg_results/Results/HalfCheetah_Policy_Act_Leaky_Relu_all_exp_rewards.npy')
hs_relu = np.load('/Users/Riashat/Documents/PhD_Research/OpenAIBaselines/ReproducibilityML/Results/rllab_results/baselines_ddpg_results/Results/HalfCheetah_Policy_Act_Relu_all_exp_rewards.npy')
hs_tanh = np.load('/Users/Riashat/Documents/PhD_Research/OpenAIBaselines/ReproducibilityML/Results/rllab_results/baselines_ddpg_results/Results/HalfCheetah_Policy_Act_TanH_all_exp_rewards.npy')

mean_hs_leaky = np.mean(hs_leaky_relu, axis=1)
mean_hs_relu = np.mean(hs_relu, axis=1)
mean_hs_tanh = np.mean(hs_tanh, axis=1)

std_hs_leaky = np.std(hs_leaky_relu, axis=1)
std_hs_relu = np.std(hs_relu, axis=1)
std_hs_tanh = np.std(hs_tanh, axis=1)


last_hs_leaky = mean_hs_leaky[-1]
last_error_hs_leaky = stats.sem(hs_leaky_relu[-1, :], axis=None, ddof=0)

print ("last_hs_leaky", last_hs_leaky)
print ("last_error_hs_leaky", last_error_hs_leaky)


last_hs_relu = mean_hs_relu[-1]
last_error_hs_relu = stats.sem(hs_relu[-1, :], axis=None, ddof=0)

print ("last_hs_relu", last_hs_relu)
print ("last_error_hs_relu", last_error_hs_relu)



last_hs_tanh = mean_hs_tanh[-1]
last_error_hs_tanh = stats.sem(hs_tanh[-1, :], axis=None, ddof=0)

print ("last_hs_tanh", last_hs_tanh)
print ("last_error_hs_tanh", last_error_hs_tanh)





#Hopper Policy Activations
ho_leaky_relu = np.load('/Users/Riashat/Documents/PhD_Research/OpenAIBaselines/ReproducibilityML/Results/rllab_results/baselines_ddpg_results/Results/Hopper_Policy_Activation_Leaky_Relu_all_exp_rewards.npy')
ho_relu = np.load('/Users/Riashat/Documents/PhD_Research/OpenAIBaselines/ReproducibilityML/Results/rllab_results/baselines_ddpg_results/Results/Hopper_Policy_Activation_Relu_all_exp_rewards.npy')
ho_tanh = np.load('/Users/Riashat/Documents/PhD_Research/OpenAIBaselines/ReproducibilityML/Results/rllab_results/baselines_ddpg_results/Results/Hopper_Policy_Activation_TanH_all_exp_rewards.npy')

mean_ho_leaky = np.mean(ho_leaky_relu, axis=1)
mean_ho_relu = np.mean(ho_relu, axis=1)
mean_ho_tanh = np.mean(ho_tanh, axis=1)

std_ho_leaky = np.std(ho_leaky_relu, axis=1)
std_ho_relu = np.std(ho_relu, axis=1)
std_ho_tanh = np.std(ho_tanh, axis=1)


last_ho_leaky = mean_ho_leaky[-1]
last_error_ho_leaky = stats.sem(ho_leaky_relu[-1, :], axis=None, ddof=0)

print ("last_ho_leaky", last_ho_leaky)
print ("last_error_ho_leaky", last_error_ho_leaky)


last_ho_relu = mean_ho_relu[-1]
last_error_ho_relu = stats.sem(ho_relu[-1, :], axis=None, ddof=0)

print ("last_ho_relu", last_ho_relu)
print ("last_error_ho_relu", last_error_ho_relu)



last_ho_tanh = mean_ho_tanh[-1]
last_error_ho_tanh = stats.sem(ho_tanh[-1, :], axis=None, ddof=0)

print ("last_ho_tanh", last_ho_tanh)
print ("last_error_ho_tanh", last_error_ho_tanh)



def multiple_plot(average_vals_list, std_dev_list, traj_list, other_labels, env_name, smoothing_window=5, no_show=False, ignore_std=False, limit=None, extra_lines=None):
    # average_vals_list - list of numpy averages
    # std_dev list - standard deviation or error
    # traj_list - list of timestep (x-axis) quantities
    # other_labels - the labels for the lines
    # Env-name the header
    # smoothing window how much to smooth using a running average.

    fig = plt.figure(figsize=(16, 8))
    # fig = plt.figure(figsize=(15, 10))
    colors = ["#1f77b4", "#ff7f0e", "#d62728", "#9467bd", "#2ca02c", "#8c564b", "#e377c2", "#bcbd22", "#17becf"]
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
    
def get_plot(stats1, stats2, stats3,  smoothing_window=5, noshow=False):
    ## Figure 1
    fig = plt.figure(figsize=(70, 40))
    rewards_smoothed_1 = pd.Series(stats1).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_2 = pd.Series(stats2).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_3 = pd.Series(stats3).rolling(smoothing_window, min_periods=smoothing_window).mean()

    cum_rwd_1, = plt.plot(eps, rewards_smoothed_1, color = "red", linewidth=2.5, label="Policy Network Activation = ReLU")    
    plt.fill_between( eps, rewards_smoothed_1 + std_hs_relu,   rewards_smoothed_1 - std_hs_relu, alpha=0.2, edgecolor='red', facecolor='red')

    cum_rwd_2, = plt.plot(eps, rewards_smoothed_2, color = "blue", linewidth=2.5, label="Policy Network Activation = TanH" )  
    plt.fill_between( eps, rewards_smoothed_2 + std_hs_tanh,   rewards_smoothed_2 - std_hs_tanh, alpha=0.2, edgecolor='blue', facecolor='blue')

    cum_rwd_3, = plt.plot(eps, rewards_smoothed_3, color = "black", linewidth=2.5, label="Policy Network Activation = Leaky ReLU" )  
    plt.fill_between( eps, rewards_smoothed_3 + std_hs_leaky,   rewards_smoothed_3 - std_hs_leaky, alpha=0.2, edgecolor='black', facecolor='black')

    plt.legend(handles=[cum_rwd_1, cum_rwd_2, cum_rwd_3], fontsize=22)
    plt.xlabel("Number of Iterations",fontsize=26)
    plt.ylabel("Average Returns", fontsize=26)
    plt.title("DDPG with HalfCheetah Environment - Actor Network Activations", fontsize=30)
  
    plt.show()

    fig.savefig('ddpg_halfcheetah_policy_activations.png')
    
    return fig


def main():
   timesteps_per_epoch = 2000
   max_timesteps = 2e6
   plot_multiple(
      [mean_ho_relu, mean_ho_tanh, mean_ho_leaky],
      [std_ho_relu, std_ho_tanh, std_ho_leaky],
      [range(0, max_timesteps, timesteps_per_epoch)]*3,
      ["relu", "tanh", "leaky_relu"],
      "HalfCheetah-v1 (DDPG, Policy Network Activation)")



if __name__ == '__main__':
    main()