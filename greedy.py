"""
Implements the greedy algorithm, considering known the distributions for the different classes. The expected reward is
given by the price * the probability of selling the item knowing the distributions

Authors: Lorenzo Di Toro
"""

import numpy as np


def calculate_expected_reward(price_matrix, distribution_matrix, configuration):
    """ calculates the expected reward for a price configuration
    :param price_matrix: price matrix
    :param distribution_matrix: matrix of the distributions for the different classes of users
    :param configuration: vector representing the configuration (arms to pull)
    :return: reward
    """
    i = 0
    reward = 0

    # Prices vector corresponding to the configuration
    prices = [price_matrix[0][configuration[0]], price_matrix[1][configuration[1]],
              price_matrix[2][configuration[2]], price_matrix[3][configuration[3]],
              price_matrix[4][configuration[4]]]

    # Gets the right probability to buy each item for the corresponding price selected
    k = 0
    probabililty = []
    for k in range(5):
        j = 0
        prob = 0
        i = 0
        for i in range(3):
            for j in range(4):
                # The probability to buy something is the sum of probabilities of the arm pulled and the higher prices
                if j + configuration[k] < 4:
                    prob += distribution_matrix[i][k][j + configuration[k]]
        probabililty.append(prob)

    # For each class
    reward = 0
    j = 0
    for j in range(5):
        reward += prices[j] * probabililty[j]

    return reward


def greedy_algorithm(price_matrix, distributions, best_configuration):
    """ calculates the best price configuration
    :param price_matrix: the matrix of all the possible prices
    :param best_configuration: this is a vector indicating the column for the chosen price for each item, must be set as
    vector of zeros when called
    :param distributions: the matrices containing the distributions of the different classes
    :return: best price configuration (array)
    """
    i = 0

    # Creates a copy of the configuration
    best_configuration2 = best_configuration[:]

    # Calculates the revenue for the lowest price point
    old_revenue = calculate_expected_reward(price_matrix, distributions, best_configuration)
    # Creates another copy
    possible_configuration = best_configuration[:]

    for i in range(5):
        possible_configuration[i] += 1
        if possible_configuration[i] == 4:
            # Skips
            possible_configuration[i] -= 1
        else:
            # Calculates the revenue for this configuration
            new_revenue = calculate_expected_reward(price_matrix, distributions, possible_configuration)
            if new_revenue > old_revenue:
                best_configuration = possible_configuration[:]
                old_revenue = new_revenue
            possible_configuration[i] -= 1

    # Checks if the algorithm is finished
    flag = 0
    k = 0
    for item in best_configuration:
        if item == best_configuration2[k]:
            k = k + 1
        else:
            flag = flag + 1
    if flag == 0:
        print("FINAL CONFIGURATION REACHED", best_configuration)
        return best_configuration
    else:
        # Recursive
        return greedy_algorithm(price_matrix, distributions, best_configuration)
