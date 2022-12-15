"""
runs the simulation for step 5

Author: Lorenzo Di Toro
"""
import numpy.random

from simulatorstep4 import Simulator4
from bandit import *
from clairvoyant import *
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
DIRICHLET = np.random.dirichlet((1, 1, 1, 1, 1, 1))

# Matrix containing 4 possible values for all the different products
PRICE_MATRIX = [PRICE_POINTS1, PRICE_POINTS2, PRICE_POINTS3, PRICE_POINTS4, PRICE_POINTS5]

# Creates the matrix representing the graph, not fixed, randomly computed
SECONDARY_PRODUCTS_PROB1 = np.random.dirichlet((1, 1, 1, 1, 1), 5)

# Runs clairvoyant algorithm using 1 iteration for each configuration (and not the average as we did in other cases)
arms = [0, 0, 0, 0, 0]
sim = Simulator4(DIRICHLET, PRICE_MATRIX, arms, 100, 200, DIST, SECONDARY_PRODUCTS_PROB1, LAMBDA)
arms, _ = clairvoyant(sim, NUMBER_RUNS)

# Once we found the optimal arms configuration we run t days with that configuration
j = 0
cumulative_rewards_clairvoyant = np.zeros(NUMBER_DAYS)  # To store each day result for the clairvoyant algorithm
for j in range(NUMBER_RUNS):
    results = np.zeros((5, 4))
    t = 0
    for t in range(NUMBER_DAYS):
        simulation = Simulator4(DIRICHLET, PRICE_MATRIX, arms, 100, 200, DIST, SECONDARY_PRODUCTS_PROB1, LAMBDA)
        results += simulation.run()
        cumulative_rewards_clairvoyant[t] += results.sum()
cumulative_rewards_clairvoyant = cumulative_rewards_clairvoyant / NUMBER_RUNS  # Gets the avg cumulative reward for
# each day
print("CLAIRVOYANT RESULTS: ", cumulative_rewards_clairvoyant[NUMBER_DAYS-1])

print("NOT OPTIMIZED UCB-1")
j = 0
cumulative_rewards_ucb1 = np.zeros(NUMBER_DAYS)
for j in range(NUMBER_RUNS):
    # Monte Carlo sampling has to be run like that, we can't run it once because the graph weights change every time
    arms = [0, 0, 0, 0, 0]  # Dummy configuration
    simulation = Simulator4(DIRICHLET, PRICE_MATRIX, arms, 100, 200, DIST, SECONDARY_PRODUCTS_PROB1, LAMBDA)
    montecarlo_matrix = simulation.monte_carlo_sampling(100)  # Can't do 1000000 iteration
    # Set bandit algorithm ucb1 parameters
    results = np.zeros((5, 4))
    samples_matrix = np.zeros((5, 4))
    realizations_matrix = np.zeros((5, 4))
    added_reward = np.zeros((5, 4))
    t = 0
    previous_configuration = None  # Has to be initialized for the first iteration
    # UCB-1 not optimized, with montecarlo sampling not accurate enough
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
print("NON OPT. UCB-1 RESULTS: ", cumulative_rewards_ucb1[NUMBER_DAYS-1])

j = 0
cumulative_rewards_ucb1_opt = np.zeros(NUMBER_DAYS)
for j in range(NUMBER_RUNS):
    # UCB-1 optimized
    t = 0
    results = np.zeros((5, 4))
    samples_matrix = np.zeros((5, 4))
    realizations_matrix = np.zeros((5, 4))
    added_reward = np.zeros((5, 4))
    for t in range(NUMBER_DAYS):
        # Selects the configuration
        arms = bandit_ucb1_step5(realizations_matrix, samples_matrix, t)
        # Runs the simulation
        simulation = Simulator4(DIRICHLET, PRICE_MATRIX, arms, 100, 200, DIST, SECONDARY_PRODUCTS_PROB1, LAMBDA)
        results += simulation.run()
        realizations_matrix = results.copy()
        cumulative_rewards_ucb1_opt[t] += results.sum()
        previous_configuration = arms.copy()  # Have to get a copy to use for the monte carlo optimization
cumulative_rewards_ucb1_opt = cumulative_rewards_ucb1_opt / NUMBER_RUNS  # Same as before
print("OPT. UCB-1 RESULTS: ", cumulative_rewards_ucb1_opt[NUMBER_DAYS-1])

print("THOMPSON")
j = 0
cumulative_rewards_ts = np.zeros(NUMBER_DAYS)
for j in range(NUMBER_RUNS):
    # Monte Carlo sampling has to be run like that, we can't run it once because the graph weights change every time
    arms = [0, 0, 0, 0, 0]  # Dummy configuration
    simulation = Simulator4(DIRICHLET, PRICE_MATRIX, arms, 100, 200, DIST, SECONDARY_PRODUCTS_PROB1, LAMBDA)
    montecarlo_matrix = simulation.monte_carlo_sampling(100)  # Can't do 1000000 iteration
    # Set bandit algorithm ts parameters
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
print("TS RESULTS: ", cumulative_rewards_ts[NUMBER_DAYS-1])

print("THOMPSON OPTIMIZED")
j = 0
cumulative_rewards_ts_opt = np.zeros(NUMBER_DAYS)
for j in range(NUMBER_RUNS):
    # Set bandit algorithm ts parameters
    realizations_matrix = np.zeros((5, 4))
    samples_matrix = np.zeros((5, 4))
    mu = np.zeros((5, 4))
    sigma = np.full((5, 4), 300000)
    t = 0
    for t in range(NUMBER_DAYS):
        # Selects the configuration
        arms = bandit_ts(mu, sigma, samples_matrix)
        # Runs the simulation
        simulation = Simulator4(DIRICHLET, PRICE_MATRIX, arms, 100, 200, DIST, SECONDARY_PRODUCTS_PROB1, LAMBDA)
        realizations_matrix += simulation.run()
        cumulative_rewards_ts_opt[t] += realizations_matrix.sum()
        # Updates the distributions parameters with the results of the day
        update_ts_step5(mu, sigma, realizations_matrix, samples_matrix, arms)
cumulative_rewards_ts_opt = cumulative_rewards_ts_opt / NUMBER_RUNS  # Same as before
print("TS OPT. RESULTS: ", cumulative_rewards_ts_opt[NUMBER_DAYS-1])

# Gets cumulative regret
cumulative_regret_ucb1 = np.zeros(NUMBER_DAYS)
cumulative_regret_ts = np.zeros(NUMBER_DAYS)
cumulative_regret_ucb1_opt = np.zeros(NUMBER_DAYS)
cumulative_regret_ts_opt = np.zeros(NUMBER_DAYS)
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
        if cumulative_rewards_clairvoyant[t] - cumulative_rewards_ucb1_opt[t] < 0:
            cumulative_regret_ucb1_opt[t] = 0
        else:
            cumulative_regret_ucb1_opt[t] = cumulative_rewards_clairvoyant[t] - cumulative_rewards_ucb1_opt[t]
        if cumulative_rewards_clairvoyant[t] - cumulative_rewards_ts_opt[t] < 0:
            cumulative_regret_ts_opt[t] = 0
        else:
            cumulative_regret_ts_opt[t] = cumulative_rewards_clairvoyant[t] - cumulative_rewards_ts_opt[t]
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
        if cumulative_rewards_clairvoyant[t] - cumulative_rewards_clairvoyant[t - 1] - (
                cumulative_rewards_ucb1_opt[t] - cumulative_rewards_ucb1_opt[t - 1]) < 0:
            cumulative_regret_ucb1_opt[t] = 0
        else:
            cumulative_regret_ucb1_opt[t] = cumulative_rewards_clairvoyant[t] - cumulative_rewards_clairvoyant[t - 1] - (
                cumulative_rewards_ucb1_opt[t] - cumulative_rewards_ucb1_opt[t - 1])
        if cumulative_rewards_clairvoyant[t] - cumulative_rewards_clairvoyant[t - 1] - (
                cumulative_rewards_ts_opt[t] - cumulative_rewards_ts_opt[t - 1]):
            cumulative_regret_ts_opt[t] = 0
        else:
            cumulative_regret_ts_opt[t] = cumulative_rewards_clairvoyant[t] - cumulative_rewards_clairvoyant[t - 1] - (
                cumulative_rewards_ts_opt[t] - cumulative_rewards_ts_opt[t - 1])

cumulative_regret_ucb1 = np.cumsum(cumulative_regret_ucb1)
cumulative_regret_ts = np.cumsum(cumulative_regret_ts)
cumulative_regret_ucb1_opt = np.cumsum(cumulative_regret_ucb1_opt)
cumulative_regret_ts_opt = np.cumsum(cumulative_regret_ts_opt)

# Plots
list_regret1 = [cumulative_regret_ucb1, cumulative_regret_ucb1_opt]
list_regret2 = [cumulative_regret_ts, cumulative_regret_ts_opt]
list_reward = [cumulative_rewards_clairvoyant, cumulative_rewards_ucb1, cumulative_rewards_ts, cumulative_rewards_ucb1_opt, cumulative_rewards_ts_opt]
plot_function(list_regret1, NUMBER_DAYS, ["ucb-1", "ucb-1 optimized"], "cumulative regret", "regret")
plot_function(list_regret2, NUMBER_DAYS, ["ts", "ts optimized"], "cumulative regret", "regret")
plot_function(list_reward, NUMBER_DAYS, ["clairvoyant", "ucb-1", "ts", "ucb-1 optimized", "ts optimized"], "cumulative reward", "reward")
