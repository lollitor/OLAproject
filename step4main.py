"""
runs the simulation for step 4

Author: Lorenzo Di Toro
"""
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
# Americans >= 30
P1 = [[0.1, 0.5, 0.3, 0.1], [0.7, 0.2, 0.1, 0], [0, 0.1, 0.6, 0.3], [0, 0.1, 0.6, 0.3], [0, 0.1, 0.3, 0.6]]
# Rest of the world < 30
P2 = [[0, 0.4, 0.5, 0.1], [0.3, 0.5, 0.2, 0], [0.4, 0.3, 0.3, 0], [0.1, 0.4, 0.5, 0], [0.1, 0.5, 0.4, 0]]
DIST = [P0, P1, P2]

# Initializes alpha ratios
DIRICHLET = np.random.dirichlet((1, 1, 1, 1, 1, 1))  # Considered unknown but constant

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

# Runs t days with the arm configuration found by clairvoyant algorithm [0, 0, 0, 1, 1]
cumulative_rewards_clairvoyant = np.zeros(NUMBER_DAYS)  # To store each day result for the clairvoyant algorithm
j = 0
for j in range(NUMBER_RUNS):
    results = np.zeros((5, 4))
    t = 0
    for t in range(NUMBER_DAYS):
        simulation = Simulator4(DIRICHLET, PRICE_MATRIX, [0, 0, 0, 1, 1], 100, 200, DIST, SECONDARY_PRODUCTS_PROB1,
                                LAMBDA)
        results += simulation.run()
        cumulative_rewards_clairvoyant[t] += results.sum()
cumulative_rewards_clairvoyant = cumulative_rewards_clairvoyant / NUMBER_RUNS  # Gets the avg cumulative reward for
# each day
print("CLAIRVOYANT RESULTS: ", cumulative_rewards_clairvoyant[NUMBER_DAYS - 1])

# Set bandit algorithm ucb1 parameters
j = 0
cumulative_rewards_ucb1 = np.zeros(NUMBER_DAYS)
for j in range(NUMBER_RUNS):
    results = np.zeros((5, 4))
    samples_matrix = np.zeros((5, 4))
    realizations_matrix = np.zeros((5, 4))
    added_reward = np.zeros((5, 4))
    t = 0
    previous_configuration = None  # Has to be initialized for the first iteration
    for t in range(NUMBER_DAYS):
        # Selects the configuration
        arms = bandit_ucb1(realizations_matrix, samples_matrix, t, montecarlo_matrix, previous_configuration,
                           added_reward)
        # Runs the simulation
        simulation = Simulator4(DIRICHLET, PRICE_MATRIX, arms, 100, 200, DIST, SECONDARY_PRODUCTS_PROB1, LAMBDA)
        results += simulation.run()
        realizations_matrix = results.copy()
        cumulative_rewards_ucb1[t] += results.sum()
        previous_configuration = arms.copy()  # Have to get a copy to use for the monte carlo optimization
cumulative_rewards_ucb1 = cumulative_rewards_ucb1 / NUMBER_RUNS  # Same as before
print("UCB-1 RESULTS: ", cumulative_rewards_ucb1[NUMBER_DAYS - 1])

print("THOMPSON")
# Set bandit algorithm ts parameters
j = 0
cumulative_rewards_ts = np.zeros(NUMBER_DAYS)
for j in range(NUMBER_RUNS):
    realizations_matrix = np.zeros((5, 4))
    samples_matrix = np.zeros((5, 4))
    mu = np.zeros((5, 4))
    sigma = np.full((5, 4), 300000)
    added_reward = np.zeros((5, 4))
    t = 0
    for t in range(NUMBER_DAYS):
        # Selects the configuration
        arms = bandit_ts(mu, sigma, samples_matrix)
        # Runs the simulation
        simulation = Simulator4(DIRICHLET, PRICE_MATRIX, arms, 100, 200, DIST, SECONDARY_PRODUCTS_PROB1, LAMBDA)
        realizations_matrix += simulation.run()
        cumulative_rewards_ts[t] += realizations_matrix.sum()
        # Updates the distributions parameters with the results of the day
        update_ts(mu, sigma, realizations_matrix, samples_matrix, arms, montecarlo_matrix,
                  added_reward)
cumulative_rewards_ts = cumulative_rewards_ts / NUMBER_RUNS  # Same as before
print("TS RESULTS: ", cumulative_rewards_ts[NUMBER_DAYS - 1])

# Cumulative regret calculation
cumulative_regret_ucb1 = np.zeros(NUMBER_DAYS)
cumulative_regret_ts = np.zeros(NUMBER_DAYS)
t = 0
for t in range(NUMBER_DAYS):
    if t == 0:
        if cumulative_rewards_clairvoyant[t] - cumulative_rewards_ucb1[t] < 0:
            cumulative_regret_ucb1[t] = 0
        else:
            cumulative_regret_ucb1[t] = cumulative_rewards_clairvoyant[t] - cumulative_rewards_ucb1[t]
        if cumulative_rewards_clairvoyant[t] - cumulative_rewards_ts[t] < 0:
            cumulative_regret_ts[t] = 0
        else:
            cumulative_regret_ts[t] = cumulative_rewards_clairvoyant[t] - cumulative_rewards_ts[t]
    else:
        if cumulative_rewards_clairvoyant[t] - cumulative_rewards_clairvoyant[t - 1] - (
                cumulative_rewards_ucb1[t] - cumulative_rewards_ucb1[t - 1]) < 0:
            cumulative_regret_ucb1[t] = 0
        else:
            cumulative_regret_ucb1[t] = cumulative_rewards_clairvoyant[t] - cumulative_rewards_clairvoyant[t - 1] - (
                    cumulative_rewards_ucb1[t] - cumulative_rewards_ucb1[t - 1])
        if cumulative_rewards_clairvoyant[t] - cumulative_rewards_clairvoyant[t - 1] - (
                cumulative_rewards_ts[t] - cumulative_rewards_ts[t - 1]) < 0:
            cumulative_regret_ts[t] = 0
        else:
            cumulative_regret_ts[t] = cumulative_rewards_clairvoyant[t] - cumulative_rewards_clairvoyant[t - 1] - (
                    cumulative_rewards_ts[t] - cumulative_rewards_ts[t - 1])
cumulative_regret_ucb1 = np.cumsum(cumulative_regret_ucb1)
cumulative_regret_ts = np.cumsum(cumulative_regret_ts)

list_reward = [cumulative_rewards_clairvoyant, cumulative_rewards_ucb1, cumulative_rewards_ts]
list_regrets = [cumulative_regret_ucb1, cumulative_regret_ts]
# Plot
plot_function(list_regrets, NUMBER_DAYS, ["ucb-1", "ts"], "cumulative regret", "regret")
plot_function(list_reward, NUMBER_DAYS, ["clairvoyant", "ucb-1", "ts"], "cumulative reward", "reward")

