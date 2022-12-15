"""
Implements Bandit algorithm in UCB1 and TS

Authors: Lorenzo Di Toro
"""

import math
import random

import numpy as np


def bandit_ucb1(realizations, number_samples_matrix, t, montecarlo, previous_arms, added_value):
    """ Bandit algorithm with upper confidence bound
    :param realizations: list of total number of user that bought for that price until now
    :param number_samples_matrix: 5x4 matrix of times that price was used (the arm was pulled)
    :param t: day
    :param montecarlo: 5x5 matrix from monte carlo sampling
    :param previous_arms: last configuration used, to add the values using the montecarlo results
    :param added_value: matrix added values
    :return: ucb1 chosen arm
    """
    arms = []

    # Calculates the expected reward for each item for each price considering the monte carlo sampling
    if previous_arms is not None:
        i = 0
        for i in range(5):
            j = 0
            added_value[i][previous_arms[i]] = 0
            for j in range(5):
                if number_samples_matrix[j][previous_arms[j]] == 0:
                    continue
                else:
                    added_value[i][previous_arms[i]] += montecarlo[i][j] * realizations[j][previous_arms[j]] / \
                                                        number_samples_matrix[j][previous_arms[j]]

    if t == 0:
        # Pull first arm
        arms = [0, 0, 0, 0, 0]
        # updates the parameters
        for i in range(5):
            number_samples_matrix[i][arms[i]] += 1
        return arms
    elif t == 1:
        # Pull second arm
        arms = [1, 1, 1, 1, 1]
        # Updates parameters
        for i in range(5):
            number_samples_matrix[i][arms[i]] += 1
        return arms
    elif t == 2:
        # Pull third arm
        arms = [2, 2, 2, 2, 2]
        # Updates parameters
        for i in range(5):
            number_samples_matrix[i][arms[i]] += 1
        return arms
    elif t == 3:
        # Pull fourth arm
        arms = [3, 3, 3, 3, 3]
        # Updates the parameters
        for i in range(5):
            number_samples_matrix[i][arms[i]] += 1
        return arms

    # Calculates upper confidence bound
    k = 0
    ucb = np.zeros((5, 4))
    for k in range(5):
        ucb = []
        flag = 0
        for i in range(4):
            if number_samples_matrix[k][i] == 0 and flag == 0:  # If this candidate was never tested in the last
                # windows days
                arms.append(i)  # Pull it in the next iteration
                flag = 1
            else:
                ucb.append((realizations[k][i] / number_samples_matrix[k][i]) + added_value[k][i] + \
                           math.sqrt(2 * math.log(t)) / (number_samples_matrix[k][i]))
        # Get arm with maximum ucb
        if flag == 0:
            arms.append(ucb.index(max(ucb)))

    # Update the parameters
    for i in range(5):
        number_samples_matrix[i][arms[i]] += 1

    return arms


def bandit_ucb1_step5(realizations, number_samples_matrix, t):
    """ Bandit algorithm with upper confidence bound
    :param realizations: list of total number of user that bought for that price until now
    :param number_samples_matrix: 5x4 matrix of times that price was used (the arm was pulled)
    :param t: day
    :return: ucb1 chosen arm
    """
    arms = []
    if t == 0:
        # Pull first arm
        arms = [0, 0, 0, 0, 0]
        # updates the parameters
        for i in range(5):
            number_samples_matrix[i][arms[i]] += 1
        return arms
    elif t == 1:
        # Pull second arm
        arms = [1, 1, 1, 1, 1]
        # Updates parameters
        for i in range(5):
            number_samples_matrix[i][arms[i]] += 1
        return arms
    elif t == 2:
        # Pull third arm
        arms = [2, 2, 2, 2, 2]
        # Updates parameters
        for i in range(5):
            number_samples_matrix[i][arms[i]] += 1
        return arms
    elif t == 3:
        # Pull fourth arm
        arms = [3, 3, 3, 3, 3]
        # Updates the parameters
        for i in range(5):
            number_samples_matrix[i][arms[i]] += 1
        return arms

    # Calculates upper confidence bound
    k = 0
    ucb = np.zeros((5, 4))
    for k in range(5):
        ucb = []
        for i in range(4):
            ucb.append((realizations[k][i] / number_samples_matrix[k][i]) + \
                       math.sqrt(2 * math.log(t)) / (number_samples_matrix[k][i]))
        # Get arm with maximum ucb
        arms.append(ucb.index(max(ucb)))

    # Update the parameters
    for i in range(5):
        number_samples_matrix[i][arms[i]] += 1

    return arms


def bandit_ucb1_change_detection(realizations, number_samples_matrix, t, montecarlo, previous_arms, added_value,
                                 change_time):
    """ Bandit algorithm with upper confidence bound
    :param realizations: list of total number of user that bought for that price until now
    :param number_samples_matrix: 5x4 matrix of times that price was used (the arm was pulled)
    :param t: day
    :param montecarlo: 5x5 matrix from monte carlo sampling
    :param previous_arms: last configuration used, to add the values using the montecarlo results
    :param added_value: matrix added values
    :return: ucb1 chosen arm
    """
    arms = []

    # Calculates the expected reward for each item for each price considering the monte carlo sampling
    if previous_arms is not None:
        i = 0
        for i in range(5):
            j = 0
            added_value[i][previous_arms[i]] = 0
            for j in range(5):
                if number_samples_matrix[j][previous_arms[j]] == 0:
                    continue
                else:
                    added_value[i][previous_arms[i]] += montecarlo[i][j] * realizations[j][previous_arms[j]] / \
                                                        number_samples_matrix[j][previous_arms[j]]

    if t == 0:
        # Pull first arm
        print("START UCB", t)
        arms = [0, 0, 0, 0, 0]
        # updates the parameters
        for i in range(5):
            number_samples_matrix[i][arms[i]] += 1
        return arms
    elif t == 1:
        # Pull second arm
        arms = [1, 1, 1, 1, 1]
        # Updates parameters
        for i in range(5):
            number_samples_matrix[i][arms[i]] += 1
        return arms
    elif t == 2:
        # Pull third arm
        arms = [2, 2, 2, 2, 2]
        # Updates parameters
        for i in range(5):
            number_samples_matrix[i][arms[i]] += 1
        return arms
    elif t == 3:
        # Pull fourth arm
        arms = [3, 3, 3, 3, 3]
        # Updates the parameters
        for i in range(5):
            number_samples_matrix[i][arms[i]] += 1
        return arms

    if t == change_time + 1:
        # Pull first arm
        print("RESTART UCB", t)
        arms = [0, 0, 0, 0, 0]
        # updates the parameters
        for i in range(5):
            number_samples_matrix[i][arms[i]] += 1
        return arms
    elif t == change_time + 2:
        # Pull second arm
        arms = [1, 1, 1, 1, 1]
        # Updates parameters
        for i in range(5):
            number_samples_matrix[i][arms[i]] += 1
        return arms
    elif t == change_time + 3:
        # Pull third arm
        arms = [2, 2, 2, 2, 2]
        # Updates parameters
        for i in range(5):
            number_samples_matrix[i][arms[i]] += 1
        return arms
    elif t == change_time + 4:
        # Pull fourth arm
        arms = [3, 3, 3, 3, 3]
        # Updates the parameters
        for i in range(5):
            number_samples_matrix[i][arms[i]] += 1
        return arms

    # Calculates upper confidence bound
    k = 0
    ucb = np.zeros((5, 4))
    for k in range(5):
        ucb = []
        flag = 0
        for i in range(4):
            if number_samples_matrix[k][i] == 0 and flag == 0:  # If this candidate was never tested in the last
                # windows days
                arms.append(i)  # Pull it in the next iteration
                flag = 1
            else:
                ucb.append((realizations[k][i] / number_samples_matrix[k][i]) + added_value[k][i] + \
                           math.sqrt(2 * math.log(t)) / (number_samples_matrix[k][i]))
        # Get arm with maximum ucb
        if flag == 0:
            arms.append(ucb.index(max(ucb)))

    # Update the parameters
    for i in range(5):
        number_samples_matrix[i][arms[i]] += 1

    return arms


def change_detection(results, sample_matrix, day_results, sigma_matrix, arms, t, last_detection):
    """ It has to get the result matrix BEFORE it is updated (the same passed to arm selection algorithm)
    :param results: same result matrix as ucb algorithm
    :param sample_matrix: same as ucb function
    :param day_results: last days results
    :param sigma_matrix: 5x4 matrix with updated sigma values
    :param arms: last arms configuration
    :param t: time
    :param last_detection: time at which the last detection was sensed
    :return: time in which change has been detected, 0 if no change detected
    """
    i = 0
    if t < 4 or t < last_detection + 5:
        return 0
    for i in range(5):
        if sigma_matrix[i][arms[i]] != 0:
            if abs((results[i][arms[i]] / (sample_matrix[i][arms[i]] - 1)) - day_results[i][arms[i]]) > 3 * \
                    sigma_matrix[i][
                        arms[i]]:
                return t
    return 0


def bandit_ts(mu, sigma, matrix_number_visits):
    """ Implements Thompson Sampling with normal distributions
    :param mu: mean average
    :param sigma: standard deviation
    :param matrix_number_visits: matrix containing number of times each candidate was pulled
    :return: arms to pull
    """
    i = 0
    arms = []
    # For each item
    for i in range(5):
        samples = []
        k = 0
        # For each candidate
        for k in range(4):
            # Draws a sample for each candidate for that price
            samples.append(np.random.normal(mu[i][k], sigma[i][k]))
        # Finds the maximum for that item
        arms.append(samples.index(max(samples)))

        # Update matrix of number of times that candidate was pulled
        matrix_number_visits[i][arms[i]] += 1

    return arms


def update_ts(mu, sigma, reward_matrix, samples_matrix, previous_arms, montecarlo, added_value):
    """ Updates avg mean and standard deviation of the normal distributions
    :param mu: avg mean matrix
    :param sigma: standard deviation matrix
    :param reward_matrix:
    :param samples_matrix:
    :param previous_arms: last arm configuration chosen
    :param montecarlo: 5x5 matrix with montecarlo sampling results
    :param added_value: 5x4 matrix that considers the expected reward generated by clicking on the first and second slot
    :return: updated mu and sigma (void)
    """
    # Calculates the expected reward for each item for each price considering the monte carlo sampling
    if previous_arms is not None:
        i = 0
        for i in range(5):
            j = 0
            added_value[i][previous_arms[i]] = 0
            for j in range(5):
                added_value[i][previous_arms[i]] += montecarlo[i][j] * reward_matrix[j][previous_arms[j]] / \
                                                    samples_matrix[j][previous_arms[j]]

    i = 0
    for i in range(5):
        # Update the normal distribution parameters for that candidate
        sigma_sqr = 1 / ((1 / pow(300000, 2)) + samples_matrix[i][previous_arms[i]])
        mu[i][previous_arms[i]] = sigma_sqr * (reward_matrix[i][previous_arms[i]] + added_value[i][previous_arms[i]])
        sigma[i][previous_arms[i]] = math.sqrt(sigma_sqr)


def update_ts_step5(mu, sigma, reward_matrix, samples_matrix, previous_arms):
    """ Updates avg mean and standard deviation of the normal distributions
    :param mu: avg mean matrix
    :param sigma: standard deviation matrix
    :param reward_matrix:
    :param samples_matrix:
    :param previous_arms: last configuration chosen
    :return: updated mu and sigma (void)
    """
    i = 0
    for i in range(5):
        # Update the normal distribution parameters for that candidate
        sigma_sqr = 1 / ((1 / pow(300000, 2)) + samples_matrix[i][previous_arms[i]])
        mu[i][previous_arms[i]] = sigma_sqr * (reward_matrix[i][previous_arms[i]])
        sigma[i][previous_arms[i]] = math.sqrt(sigma_sqr)


def regret_upper_bound_ucb1(clairvoyant_result, ucb_results, number_days):
    """ calculates regret theoretical upper bound for ucb-1
    :param clairvoyant_result: clairvoyant cumulative reward
    :param ucb_results: algorithm cumulative reward
    :param number_days: number of days simulated
    :return: the upper bound of the regret
    """
    upper_bound = ((4 * math.log(number_days, 10)) / (clairvoyant_result - ucb_results)) + 8 * (
            clairvoyant_result - ucb_results)
    return upper_bound


def regret_upper_bound_ts(clairvoyant_result, ts_results, number_days):
    """ calculates regret theoretical upper bound for ts
    :param clairvoyant_result: clairvoyant cumulative reward
    :param ts_results: algorithm cumulative reward
    :param number_days: number of days simulated
    :return: the upper bound of the regret
    """
    upper_bound = (clairvoyant_result - ts_results) * (
                math.log(number_days, 10) + math.log(math.log(number_days, 10), 10)) / (
                              clairvoyant_result * math.log(clairvoyant_result / ts_results, 10))
    return upper_bound
