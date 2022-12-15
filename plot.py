"""
Function that plots given two vectors

Authors: Lorenzo Di Toro, ...
"""
import numpy as np
from matplotlib import pyplot as plt


def plot_function(list_vectors, n_days, names_list, graph_name, y_name):
    """ plots the vector1 with vector 2 as standard deviation
    :param list_vectors: list of lists representing the functions to plot
    :param n_days: number of days
    :param names_list: name list for each function in the list
    :param graph_name: name of the graph
    :param y_name: label of the y axis
    :return: shows the plot
    """
    # Gets std deviation given the vector and the number of days
    j = 0
    std_deviations = np.zeros((len(list_vectors), n_days))
    for j in range(len(list_vectors)):
        t = 0
        for t in range(n_days):
            if t == 0:
                std_deviations[j][t] = 0
            else:
                std_deviations[j][t] = np.std(list_vectors[j][:t])

    x = list(range(0, n_days))
    plt.title(graph_name)
    plt.xlabel("t")
    plt.ylabel(y_name)
    j = 0
    for j in range(len(list_vectors)):
        plt.plot(x, list_vectors[j], label=names_list[j])
        plt.fill_between(x, (list_vectors[j] - std_deviations[j]), (list_vectors[j] + std_deviations[j]), alpha=0.3)
    plt.legend()
    plt.show()
