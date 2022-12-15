"""
runs the simulation for step 6

Author: Lorenzo Di Toro
"""
import math

import numpy.random

from simulatorstep4 import Simulator4
from bandit import *
from plot import plot_function

# Creates the constant to pass
NUMBER_DAYS = 300
NUMBER_RUNS = 10
LAMBDA = 0.5
PRODUCTS = [0, 1, 2, 3, 4]
PRICE_POINTS1 = [350, 400, 450, 500]  # Playstation 5
PRICE_POINTS2 = [250, 300, 350, 400]  # Playstation 4
PRICE_POINTS3 = [400, 450, 500, 550]  # Xbox series X
PRICE_POINTS4 = [200, 250, 300, 350]  # Xbox series S
PRICE_POINTS5 = [600, 800, 1000, 1200]  # Gaming PC

# Distributions
# Americans < 30
P0 = [[0.6, 0.2, 0.1, 0.1], [0.7, 0.2, 0.1, 0], [0.1, 0.2, 0.6, 0.1], [0, 0.1, 0.6, 0.3], [0.6, 0.2, 0.1, 0.1]]
P0_2 = [[0.4, 0.4, 0.1, 0.1], [0.5, 0.4, 0.1, 0], [0.1, 0.4, 0.4, 0.1], [0, 0.4, 0.3, 0.3], [0.6, 0.2, 0.1, 0.1]]
# Americans >= 30
P1 = [[0.1, 0.5, 0.3, 0.1], [0.7, 0.2, 0.1, 0], [0, 0.1, 0.6, 0.3], [0, 0.1, 0.6, 0.3], [0, 0.1, 0.3, 0.6]]
P1_2 = [[0.1, 0.3, 0.5, 0.1], [0.3, 0.2, 0.5, 0], [0.3, 0.1, 0.3, 0.3], [0, 0.5, 0.2, 0.3], [0.5, 0.1, 0.3, 0.1]]
# Rest of the world < 30
P2 = [[0, 0.4, 0.5, 0.1], [0.3, 0.5, 0.2, 0], [0.4, 0.3, 0.3, 0], [0.1, 0.4, 0.5, 0], [0.1, 0.5, 0.4, 0]]
P2_2 = [[0.2, 0.2, 0.5, 0.1], [0.1, 0.7, 0.2, 0], [0.1, 0.6, 0.3, 0], [0, 0.4, 0.5, 0.1], [0, 0.6, 0.4, 0]]
DIST = [P0, P1, P2]
DIST_2 = [P0_2, P1_2, P2_2]  # Distribution for the abrupt change

# Initializes alpha ratios
DIRICHLET = np.random.dirichlet((1, 1, 1, 1, 1, 1))

# Matrix containing 4 possible values for all the different products
PRICE_MATRIX = [PRICE_POINTS1, PRICE_POINTS2, PRICE_POINTS3, PRICE_POINTS4, PRICE_POINTS5]

# Creates the matrix representing the graph
SECONDARY_PRODUCTS_PROB1 = [[0, 0.2, 0.3, 0.1, 0.4], [0.2, 0, 0.3, 0.1, 0.4], [0.3, 0.2, 0, 0.1, 0.4],
                            [0.1, 0.2, 0.3, 0, 0.4], [0.4, 0.2, 0.3, 0.1, 0]]

# Monte Carlo sampling results got from 1000000 iteration on the Simulator4
montecarlo_matrix = [[0., 0.04610762, 0.06914527, 0.02310574, 0.09233486],
                     [0.04611269, 0., 0.06919115, 0.02314037, 0.09242766],
                     [0.06905776, 0.04613647, 0., 0.02304631, 0.09240796],
                     [0.02308017, 0.04623074, 0.06909782, 0., 0.09230339],
                     [0.09229981, 0.04624623, 0.06922937, 0.02312042, 0.]]

j = 0
cumulative_rewards_clairvoyant = np.zeros(NUMBER_DAYS)
for j in range(NUMBER_RUNS):
    # Runs t days with the arm configuration found by clairvoyant algorithm [0, 0, 0, 1, 1]
    results = np.zeros((5, 4))
    t = 0
    for t in range(NUMBER_DAYS):
        if t < int(NUMBER_DAYS / 2):
            simulation = Simulator4(DIRICHLET, PRICE_MATRIX, [0, 0, 0, 1, 1], 100, 200, DIST, SECONDARY_PRODUCTS_PROB1,
                                    LAMBDA)
            results += simulation.run()
        else:
            simulation = Simulator4(DIRICHLET, PRICE_MATRIX, [0, 1, 1, 1, 1], 100, 200, DIST_2,
                                    SECONDARY_PRODUCTS_PROB1,
                                    LAMBDA)
            results += simulation.run()
        cumulative_rewards_clairvoyant[t] += results.sum()
cumulative_rewards_clairvoyant = cumulative_rewards_clairvoyant / NUMBER_RUNS  # Gets the avg cumulative reward for
# each day
print("CLAIRVOYANT RESULTS: ", cumulative_rewards_clairvoyant[NUMBER_DAYS - 1])

x = 0
cumulative_rewards_ucb_sw = np.zeros(NUMBER_DAYS)
window_size = int(math.sqrt(NUMBER_DAYS)) * 2
for x in range(NUMBER_RUNS):
    # Set bandit algorithm ucb1 parameters with sliding window
    results = np.zeros((5, 4))
    t = 0
    window = np.zeros((window_size, 5, 4))
    window_expected = np.zeros((window_size, 5, 4))
    window_samples = np.zeros((window_size, 5, 4))
    previous_configuration = None  # Has to be initialized for the first iteration
    window_sum = np.zeros((5, 4))  # Create the matrix for the sum of the last 60 days
    samples_sum = np.zeros((5, 4))
    expected_sum = np.zeros((5, 4))
    day_samples = np.zeros((5, 4))
    day_expected = np.zeros((5, 4))
    previous_day_samples = np.zeros((5, 4))
    previous_day_expected = np.zeros((5, 4))
    for t in range(NUMBER_DAYS):
        # Selects the configuration with a sliding window
        i = t % window_size
        j = 0
        window_sum = np.zeros((5, 4))
        samples_sum = np.zeros((5, 4))
        expected_sum = np.zeros((5, 4))
        # Sums the last 60 daily results
        for j in range(window_size):
            window_sum += window[j]
            samples_sum += window_samples[j]
            expected_sum += window_expected[j]
        # Stores the result of the samples and the added reward to get the daily difference
        previous_day_samples = samples_sum.copy()
        previous_day_expected = expected_sum.copy()
        # Picks the arm configuration and updates the samples and added reward matrix
        arms = bandit_ucb1(window_sum, samples_sum, t, montecarlo_matrix,
                           previous_configuration,
                           expected_sum)
        # Gets the daily difference
        day_samples = samples_sum - previous_day_samples
        day_expected = expected_sum - previous_day_expected
        # Puts the daily difference in the list of matrices
        window_samples[i] = day_samples.copy()
        window_expected[i] = day_expected.copy()
        # Runs the simulation
        if t < int(NUMBER_DAYS / 2):
            simulation = Simulator4(DIRICHLET, PRICE_MATRIX, arms, 100, 200, DIST, SECONDARY_PRODUCTS_PROB1, LAMBDA)
        else:
            # Abrupt change of demand curves
            simulation = Simulator4(DIRICHLET, PRICE_MATRIX, arms, 100, 200, DIST_2, SECONDARY_PRODUCTS_PROB1, LAMBDA)
        # Gets the daily award
        window[i] = simulation.run()
        results += window[i]  # Gets cumulative reward
        cumulative_rewards_ucb_sw[t] += results.sum()
        previous_configuration = arms.copy()  # Have to get a copy to use for the monte carlo optimization
cumulative_rewards_ucb_sw = cumulative_rewards_ucb_sw / NUMBER_RUNS

x = 0
cumulative_rewards_ucb_cp = np.zeros(NUMBER_DAYS)
for x in range(NUMBER_RUNS):
    # UCB-1 with demand curve abrupt change detection
    results = np.zeros((5, 4))
    t = 0
    change_time = None
    samples_matrix = np.zeros((5, 4))
    expected_matrix = np.zeros((5, 4))
    time_matrix = np.zeros((5, 4))  # Matrix that tracks last time when an arm was pulled
    previous_configuration = None  # Has to be initialized for the first iteration
    sigma_matrix = np.zeros((5, 4))  # Create the matrix that stores the confidence bounds for each candidate
    results_list = []
    day_results = np.zeros((5, 4))
    for t in range(NUMBER_DAYS):
        # Picks the arm configuration and updates the samples and added reward matrix
        arms = bandit_ucb1_change_detection(results, samples_matrix, t, montecarlo_matrix,
                                            previous_configuration,
                                            expected_matrix, change_time)
        # Runs the simulation
        if t < int(NUMBER_DAYS / 2):
            simulation = Simulator4(DIRICHLET, PRICE_MATRIX, arms, 100, 200, DIST, SECONDARY_PRODUCTS_PROB1, LAMBDA)
        else:
            # Abrupt change of demand curves
            simulation = Simulator4(DIRICHLET, PRICE_MATRIX, arms, 100, 200, DIST_2, SECONDARY_PRODUCTS_PROB1, LAMBDA)
        # Gets the daily award
        day_results = simulation.run()
        results_list.append(day_results.copy())  # Append the results to calculate the standard deviation
        # Sigma calculation
        i = 0
        j = 0
        k = 0
        # For each element
        for i in range(5):
            for j in range(4):
                element_results = []
                # For each matrix
                for k in range(len(results_list)):
                    element_results.append(results_list[k][i][j])
                sigma_matrix[i][j] = np.std(element_results)  # We create a list of the same element through all the
                # matrices. Than we calculate the sigma value for each vector
        change_time = change_detection(results, samples_matrix, day_results, sigma_matrix, arms, t, change_time)
        results += day_results  # Gets cumulative reward
        cumulative_rewards_ucb_cp[t] += results.sum()
        previous_configuration = arms.copy()  # Have to get a copy to use for the monte carlo optimization
cumulative_rewards_ucb_cp = cumulative_rewards_ucb_cp / NUMBER_RUNS

# Cumulative regret calculation
cumulative_regret_ucb_sw = np.zeros(NUMBER_DAYS)
cumulative_regret_ucb_cp = np.zeros(NUMBER_DAYS)
t = 0
for t in range(NUMBER_DAYS):
    if t == 0:
        if cumulative_rewards_clairvoyant[t] - cumulative_rewards_ucb_sw[t] < 0:
            cumulative_regret_ucb_sw[t] = 0
        else:
            cumulative_regret_ucb_sw[t] = cumulative_rewards_clairvoyant[t] - cumulative_rewards_ucb_sw[t]
        if cumulative_rewards_clairvoyant[t] - cumulative_rewards_ucb_cp[t] < 0:
            cumulative_regret_ucb_cp[t] = 0
        else:
            cumulative_regret_ucb_cp[t] = cumulative_rewards_clairvoyant[t] - cumulative_rewards_ucb_cp[t]
    else:
        if cumulative_rewards_clairvoyant[t] - cumulative_rewards_clairvoyant[t - 1] - (
                cumulative_rewards_ucb_sw[t] - cumulative_rewards_ucb_sw[t - 1]) < 0:
            cumulative_regret_ucb_sw[t] = 0
        else:
            cumulative_regret_ucb_sw[t] = cumulative_rewards_clairvoyant[t] - cumulative_rewards_clairvoyant[t - 1] - (
                cumulative_rewards_ucb_sw[t] - cumulative_rewards_ucb_sw[t - 1])
        if cumulative_rewards_clairvoyant[t] - cumulative_rewards_clairvoyant[t - 1] - (
                cumulative_rewards_ucb_cp[t] - cumulative_rewards_ucb_cp[t - 1]) < 0:
            cumulative_regret_ucb_cp[t] = 0
        else:
            cumulative_regret_ucb_cp[t] = cumulative_rewards_clairvoyant[t] - cumulative_rewards_clairvoyant[t - 1] - (
                cumulative_rewards_ucb_cp[t] - cumulative_rewards_ucb_cp[t - 1])

cumulative_regret_ucb_sw = np.cumsum(cumulative_regret_ucb_sw)
cumulative_regret_ucb_cp = np.cumsum(cumulative_regret_ucb_cp)

list_regrets = [cumulative_regret_ucb_sw, cumulative_regret_ucb_cp]
list_rewards = [cumulative_rewards_clairvoyant, cumulative_rewards_ucb_sw, cumulative_rewards_ucb_cp]
plot_function(list_regrets, NUMBER_DAYS, ["ucb sw", "ucb changepoint"], "cumulative regret", "regret")
plot_function(list_rewards, NUMBER_DAYS, ["clairvoyant", "ucb sw", "ucb changepoint"], "cumulative reward", "reward")
