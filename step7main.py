"""
runs the simulation for step 7

Author: Lorenzo Di Toro
"""
import numpy.random

from simulatorfirstclass import SimulatorFirst
from simulatorsecondclass import SimulatorSecond
from simulatorthirdclass import SimulatorThird
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
        # Finds number of users to pass to the simulation for each class
        number_users = random.randint(100, 200)
        i = 0
        first_class = 0
        second_class = 0
        third_class = 0
        for i in range(number_users):
            type_ = np.random.choice([0, 1, 2], p=[0.2, 0.1, 0.7])  # Classes decided with this distribution
            if type_ == 0:
                first_class += 1
            if type_ == 1:
                second_class += 1
            if type_ == 2:
                third_class += 1
        simulation1 = SimulatorFirst(DIRICHLET, PRICE_MATRIX, [0, 0, 1, 2, 0], first_class, DIST,
                                     SECONDARY_PRODUCTS_PROB1,
                                     LAMBDA)
        simulation2 = SimulatorSecond(DIRICHLET, PRICE_MATRIX, [1, 0, 0, 1, 1], second_class, DIST,
                                      SECONDARY_PRODUCTS_PROB1,
                                      LAMBDA)
        simulation3 = SimulatorThird(DIRICHLET, PRICE_MATRIX, [1, 0, 1, 2, 2], third_class, DIST,
                                     SECONDARY_PRODUCTS_PROB1,
                                     LAMBDA)
        results += simulation1.run()
        results += simulation2.run()
        results += simulation3.run()
        cumulative_rewards_clairvoyant[t] += results.sum()
cumulative_rewards_clairvoyant = cumulative_rewards_clairvoyant / NUMBER_RUNS  # Gets the avg cumulative reward for
# each day
print("CLAIRVOYANT RESULTS: ", cumulative_rewards_clairvoyant[NUMBER_DAYS - 1])

j = 0
cumulative_rewards_ucb1_class = np.zeros(NUMBER_DAYS)
for j in range(NUMBER_RUNS):
    # Set bandit algorithm ucb1 parameters
    t = 0
    results1 = np.zeros((5, 4))
    results2 = np.zeros((5, 4))
    results3 = np.zeros((5, 4))
    samples_matrix1 = np.zeros((5, 4))
    samples_matrix2 = np.zeros((5, 4))
    samples_matrix3 = np.zeros((5, 4))
    realizations_matrix1 = np.zeros((5, 4))
    realizations_matrix2 = np.zeros((5, 4))
    realizations_matrix3 = np.zeros((5, 4))
    added_reward1 = np.zeros((5, 4))
    added_reward2 = np.zeros((5, 4))
    added_reward3 = np.zeros((5, 4))
    previous_configuration1 = None  # Has to be initialized for the first iteration
    previous_configuration2 = None
    previous_configuration3 = None
    # UCB-1 for first class
    for t in range(NUMBER_DAYS):
        # Finds number of users to pass to the simulation for each class
        number_users = random.randint(100, 200)
        i = 0
        first_class = 0
        second_class = 0
        third_class = 0
        for i in range(number_users):
            type_ = np.random.choice([0, 1, 2], p=[0.2, 0.1, 0.7])  # Classes decided with this distribution
            if type_ == 0:
                first_class += 1
            if type_ == 1:
                second_class += 1
            if type_ == 2:
                third_class += 1
        # Selects the configuration
        arms_first = bandit_ucb1(realizations_matrix1, samples_matrix1, t, montecarlo_matrix, previous_configuration1,
                                 added_reward1)
        arms_second = bandit_ucb1(realizations_matrix2, samples_matrix2, t, montecarlo_matrix, previous_configuration2,
                                  added_reward2)
        arms_third = bandit_ucb1(realizations_matrix3, samples_matrix3, t, montecarlo_matrix, previous_configuration3,
                                 added_reward3)
        print("chosen configuration: ", arms_first, arms_second, arms_third)
        # Runs the simulation
        simulation1 = SimulatorFirst(DIRICHLET, PRICE_MATRIX, arms_first, first_class, DIST, SECONDARY_PRODUCTS_PROB1,
                                     LAMBDA)
        simulation2 = SimulatorSecond(DIRICHLET, PRICE_MATRIX, arms_second, second_class, DIST,
                                      SECONDARY_PRODUCTS_PROB1,
                                      LAMBDA)
        simulation3 = SimulatorThird(DIRICHLET, PRICE_MATRIX, arms_third, third_class, DIST, SECONDARY_PRODUCTS_PROB1,
                                     LAMBDA)
        results1 += simulation1.run()
        results2 += simulation2.run()
        results3 += simulation3.run()
        realizations_matrix1 = results1.copy()
        realizations_matrix2 = results2.copy()
        realizations_matrix3 = results3.copy()
        cumulative_rewards_ucb1_class[
            t] += realizations_matrix1.sum() + realizations_matrix2.sum() + realizations_matrix3.sum()
        previous_configuration1 = arms_first.copy()  # Have to get a copy to use for the monte carlo optimization
        previous_configuration2 = arms_second.copy()
        previous_configuration3 = arms_third.copy()
cumulative_rewards_ucb1_class = cumulative_rewards_ucb1_class / NUMBER_RUNS

j = 0
cumulative_rewards_ts_class = np.zeros(NUMBER_DAYS)
for j in range(NUMBER_RUNS):
    # Thompson algorithm
    print("THOMPSON")
    # Set bandit algorithm ts parameters
    realizations_matrix1 = np.zeros((5, 4))
    realizations_matrix2 = np.zeros((5, 4))
    realizations_matrix3 = np.zeros((5, 4))
    samples_matrix1 = np.zeros((5, 4))
    samples_matrix2 = np.zeros((5, 4))
    samples_matrix3 = np.zeros((5, 4))
    mu1 = np.zeros((5, 4))
    mu2 = np.zeros((5, 4))
    mu3 = np.zeros((5, 4))
    sigma1 = np.full((5, 4), 300000)
    sigma2 = np.full((5, 4), 300000)
    sigma3 = np.full((5, 4), 300000)
    added_reward1 = np.zeros((5, 4))
    added_reward2 = np.zeros((5, 4))
    added_reward3 = np.zeros((5, 4))
    t = 0
    for t in range(NUMBER_DAYS):
        # Gets how many user distributed how
        number_users = random.randint(100, 200)
        i = 0
        first_class = 0
        second_class = 0
        third_class = 0
        for i in range(number_users):
            type_ = np.random.choice([0, 1, 2], p=[0.2, 0.1, 0.7])  # Classes decided with this distribution
            if type_ == 0:
                first_class += 1
            if type_ == 1:
                second_class += 1
            if type_ == 2:
                third_class += 1
        # Selects the configuration
        arms_first = bandit_ts(mu1, sigma1, samples_matrix1)
        arms_second = bandit_ts(mu2, sigma2, samples_matrix2)
        arms_third = bandit_ts(mu3, sigma3, samples_matrix3)
        # Runs the simulation
        simulation1 = SimulatorFirst(DIRICHLET, PRICE_MATRIX, arms_first, first_class, DIST, SECONDARY_PRODUCTS_PROB1,
                                     LAMBDA)
        simulation2 = SimulatorSecond(DIRICHLET, PRICE_MATRIX, arms_second, second_class, DIST,
                                      SECONDARY_PRODUCTS_PROB1,
                                      LAMBDA)
        simulation1 = SimulatorFirst(DIRICHLET, PRICE_MATRIX, arms_third, third_class, DIST, SECONDARY_PRODUCTS_PROB1,
                                     LAMBDA)
        realizations_matrix1 += simulation1.run()
        realizations_matrix2 += simulation2.run()
        realizations_matrix3 += simulation3.run()
        cumulative_rewards_ts_class[
            t] += realizations_matrix1.sum() + realizations_matrix2.sum() + realizations_matrix3.sum()
        # Updates the distributions parameters with the results of the day
        update_ts(mu1, sigma1, realizations_matrix1, samples_matrix1, arms_first, montecarlo_matrix,
                  added_reward1)
        update_ts(mu2, sigma2, realizations_matrix2, samples_matrix2, arms_second, montecarlo_matrix,
                  added_reward2)
        update_ts(mu3, sigma3, realizations_matrix3, samples_matrix3, arms_third, montecarlo_matrix,
                  added_reward3)
cumulative_rewards_ts_class = cumulative_rewards_ts_class / NUMBER_RUNS

# We also run the ucb and ts algorithms for aggregate demand curve to compare results
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
cumulative_regret_ucb1_class = np.zeros(NUMBER_DAYS)
cumulative_regret_ts_class = np.zeros(NUMBER_DAYS)
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
        if cumulative_rewards_clairvoyant[t] - cumulative_rewards_ucb1_class[t] < 0:
            cumulative_regret_ucb1_class[t] = 0
        else:
            cumulative_regret_ucb1_class[t] = cumulative_rewards_clairvoyant[t] - cumulative_rewards_ucb1_class[t]
        if cumulative_rewards_clairvoyant[t] - cumulative_rewards_ts_class[t] < 0:
            cumulative_regret_ts_class[t] = 0
        else:
            cumulative_regret_ts_class[t] = cumulative_rewards_clairvoyant[t] - cumulative_rewards_ts_class[t]
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
                cumulative_rewards_ucb1_class[t] - cumulative_rewards_ucb1_class[t - 1]) < 0:
            cumulative_regret_ucb1_class[t] = 0
        else:
            cumulative_regret_ucb1_class[t] = cumulative_rewards_clairvoyant[t] - cumulative_rewards_clairvoyant[
                t - 1] - (
                                                      cumulative_rewards_ucb1_class[t] - cumulative_rewards_ucb1_class[
                                                  t - 1])
        if cumulative_rewards_clairvoyant[t] - cumulative_rewards_clairvoyant[t - 1] - (
                cumulative_rewards_ts_class[t] - cumulative_rewards_ts_class[t - 1]) < 0:
            cumulative_regret_ts_class[t] = 0
        else:
            cumulative_regret_ts_class[t] = cumulative_rewards_clairvoyant[t] - cumulative_rewards_clairvoyant[
                t - 1] - (
                                                    cumulative_rewards_ts_class[t] - cumulative_rewards_ts_class[t - 1])

cumulative_regret_ucb1 = np.cumsum(cumulative_regret_ucb1)
cumulative_regret_ts = np.cumsum(cumulative_regret_ts)
cumulative_regret_ucb1_class = np.cumsum(cumulative_regret_ucb1_class)
cumulative_regret_ts_class = np.cumsum(cumulative_regret_ts_class)

list_regrets1 = [cumulative_regret_ucb1, cumulative_regret_ucb1_class]
list_regrets2 = [cumulative_regret_ts, cumulative_regret_ts_class]
list_reward = [cumulative_rewards_clairvoyant, cumulative_rewards_ucb1, cumulative_rewards_ts,
               cumulative_rewards_ucb1_class, cumulative_rewards_ts_class]
# Plot
plot_function(list_regrets1, NUMBER_DAYS, ["ucb-1 aggregated", "ucb-1 by class"], "cumulative regret", "regret")
plot_function(list_regrets2, NUMBER_DAYS, ["ts aggregated", "ts by class"], "cumulative regret", "regret")
plot_function(list_reward, NUMBER_DAYS,
              ["clairvoyant", "ucb-1 aggregated", "ts aggregated", "ucb-1 by class", "ts by class"],
              "cumulative regret", "regret")
