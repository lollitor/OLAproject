"""
runs the simulation for step 6

Author: Lorenzo Di Toro
"""
import numpy.random

from simulatorstep4 import Simulator4
from bandit import *

# Creates the constant to pass
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
montecarlo_matrix = [[0., 0.04637603, 0.0690013, 0.02307224, 0.09224509],
                     [0.04607599, 0., 0.0694188, 0.0230968, 0.09264245],
                     [0.06919698, 0.04651311, 0., 0.0230654, 0.09199315],
                     [0.02290587, 0.04636638, 0.06917147, 0., 0.0922835],
                     [0.0923754, 0.04631763, 0.06928731, 0.02318572, 0.]]

# Runs t days with the arm configuration found by clairvoyant algorithm [0, 0, 0, 1, 1]
results = np.zeros((5, 4))
t = 0
for t in range(150):
    simulation = Simulator4(DIRICHLET, PRICE_MATRIX, [0, 0, 0, 1, 1], 100, 200, DIST, SECONDARY_PRODUCTS_PROB1, LAMBDA)
    results += simulation.run()

# Runs t days with the arm configuration found by clairvoyant algorithm [0, 0, 1, 1, 1] after curve change
for t in range(150):
    simulation = Simulator4(DIRICHLET, PRICE_MATRIX, [0, 0, 1, 1, 1], 100, 200, DIST_2, SECONDARY_PRODUCTS_PROB1,
                            LAMBDA)
    results += simulation.run()
print("CLAIRVOYANT RESULTS: ", results.sum())

# Set bandit algorithm ucb1 parameters with sliding window
results = np.zeros((5, 4))
t = 0
window = np.zeros((17, 5, 4))
window_expected = np.zeros((17, 5, 4))
window_samples = np.zeros((17, 5, 4))
previous_configuration = None  # Has to be initialized for the first iteration
window_sum = np.zeros((5, 4))  # Create the matrix for the sum of the last 60 days
samples_sum = np.zeros((5, 4))
expected_sum = np.zeros((5, 4))
day_samples = np.zeros((5, 4))
day_expected = np.zeros((5, 4))
previous_day_samples = np.zeros((5, 4))
previous_day_expected = np.zeros((5, 4))
for t in range(150):
    # Selects the configuration with a sliding window
    i = t % 17  # Sliding window of 30 days
    j = 0
    window_sum = np.zeros((5, 4))
    samples_sum = np.zeros((5, 4))
    expected_sum = np.zeros((5, 4))
    # Sums the last 60 daily results
    for j in range(17):
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
    if t < 150:
        simulation = Simulator4(DIRICHLET, PRICE_MATRIX, arms, 100, 200, DIST, SECONDARY_PRODUCTS_PROB1, LAMBDA)
    else:
        # Abrupt change of demand curves
        simulation = Simulator4(DIRICHLET, PRICE_MATRIX, arms, 100, 200, DIST_2, SECONDARY_PRODUCTS_PROB1, LAMBDA)
    # Gets the daily award
    window[i] = simulation.run()
    results += window[i]  # Gets cumulative reward
    previous_configuration = arms.copy()  # Have to get a copy to use for the monte carlo optimization
    print(results.sum())
