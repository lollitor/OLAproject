"""
runs the simulation

Author: Lorenzo Di Toro
"""
from simulatorstep2 import Simulator2

from greedy import greedy_algorithm
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
DIRICHLET = np.random.dirichlet((1, 1, 1, 1, 1, 1))

# Matrix containing 4 possible values for all the different products
PRICE_MATRIX = [PRICE_POINTS1, PRICE_POINTS2, PRICE_POINTS3, PRICE_POINTS4, PRICE_POINTS5]

# Creates the matrix representing the graph
SECONDARY_PRODUCTS_PROB1 = np.random.dirichlet((1, 1, 1, 1, 1), 5)

# Runs t days with the arm configuration found by clairvoyant algorithm [0, 0, 0, 1, 1]
cumulative_rewards_clairvoyant = np.zeros(NUMBER_DAYS)  # To store each day result for the clairvoyant algorithm
j = 0
for j in range(NUMBER_RUNS):
    results = np.zeros((5, 4))
    t = 0
    for t in range(NUMBER_DAYS):
        simulation = Simulator2(DIRICHLET, PRICE_MATRIX, [0, 0, 0, 1, 1], 200, DIST, SECONDARY_PRODUCTS_PROB1, LAMBDA)
        results += simulation.run()
        cumulative_rewards_clairvoyant[t] += results.sum()
cumulative_rewards_clairvoyant = cumulative_rewards_clairvoyant / NUMBER_RUNS  # Gets the avg cumulative reward for each day
print("CLAIRVOYANT RESULTS: ", cumulative_rewards_clairvoyant[NUMBER_DAYS - 1])

j = 0
cumulative_rewards_greedy = np.zeros(NUMBER_DAYS)
for j in range(NUMBER_RUNS):
    # Set greedy algorithm parameters
    arms = greedy_algorithm(PRICE_MATRIX, DIST, [0, 0, 0, 0, 0])
    realizations_matrix = np.zeros((5, 4))
    t = 1
    for t in range(NUMBER_DAYS):
        # Selects the configuration
        arms = greedy_algorithm(PRICE_MATRIX, DIST, arms)
        print("chosen configuration: ", arms)
        # Runs the simulation
        simulation = Simulator2(DIRICHLET, PRICE_MATRIX, arms, 200, DIST, SECONDARY_PRODUCTS_PROB1, LAMBDA)
        realizations_matrix += simulation.run()
        cumulative_rewards_greedy[t] += realizations_matrix.sum()
cumulative_rewards_greedy = cumulative_rewards_greedy / NUMBER_RUNS

# Calculates avg regret and std deviation
cumulative_regret_greedy = np.zeros(NUMBER_DAYS)
t = 0
for t in range(NUMBER_DAYS):
    if t == 0:
        if cumulative_rewards_clairvoyant[t] - cumulative_rewards_greedy[t] < 0:
            cumulative_regret_greedy[t] = 0
        else:
            cumulative_regret_greedy[t] = cumulative_rewards_clairvoyant[t] - cumulative_rewards_greedy[t]
    else:
        if cumulative_rewards_clairvoyant[t] - cumulative_rewards_clairvoyant[t - 1] - (
                cumulative_rewards_greedy[t] - cumulative_rewards_greedy[t - 1]) < 0:
            cumulative_regret_greedy[t] = 0
        else:
            cumulative_regret_greedy[t] = cumulative_rewards_clairvoyant[t] - cumulative_rewards_clairvoyant[t - 1] - (
                        cumulative_rewards_greedy[t] - cumulative_rewards_greedy[t - 1])
cumulative_regret_greedy = np.cumsum(cumulative_regret_greedy)

list_regrets = [cumulative_regret_greedy]
list_rewards = [cumulative_rewards_clairvoyant, cumulative_rewards_greedy]
plot_function(list_regrets, NUMBER_DAYS, ["greedy"], "cumulative regret", "regret")
plot_function(list_rewards, NUMBER_DAYS, ["clairvoyant", "greedy"], "cumulative reward", "reward")
