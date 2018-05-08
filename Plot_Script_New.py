"""
Taken from : Andre Cianflone
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_data_smooth(data_x, data_y, names, xlabel, ylabel, location,\
                                                            smoothing_window):
    """
    Plot mean curve on top of shaded high variance curve, tensorboard style
    Args:
        data_x: x-axis data, (list of np arrays)
        data_y: list of np arrays, y-axis
        names: your curve labels (list of strings)
        xlabel: x-axis title
        ylabel: y-axis title
        location: legend location
        smoothing_window: mean over how many data points
    """
    for i, y in enumerate(data_x):
        p = plt.plot(data_x[i], data_y[i], '-', alpha=0.25, markersize=1,label='_nolegend_')
        c = p[0].get_color()
        df = pd.DataFrame(data_y[i])
        smoothed = df[0].rolling(smoothing_window).mean()
        plt.plot(data_x[i], smoothed, '-',markersize=0.5, label=names[i],color=c)
    plt.legend(loc=location, prop={'size': 12}, numpoints=3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
def example():
    # Generate data
    size = 3000
    eps = 0.01
    x = np.linspace(0, 1.5*np.pi, size)
    t = np.linspace(0, size, size)
    y1 = np.log(x+eps)+np.random.random(size)*.6
    sin = np.sin(x)+np.random.random(size)*4
    y2 = 0.5*sin + 0.5*y1

    # Plotting arguments
    data_x = [t, t]
    data_y = [y1, y2]
    names=["curve 1", "curve 2"]
    xlabel="time steps"
    ylabel="average return"
    location="lower right"
    window = 15

    # Plot
    plt.clf()
    plt.figure(figsize=(10,5))
    plot_data_smooth(data_x, data_y, names, xlabel, ylabel, location, window)
    plt.savefig('example.png')
example()
