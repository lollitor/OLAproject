"""
runs the clairvoyant algorithm

Author: Lorenzo Di Toro
"""
import numpy.random

from clairvoyant import *
from simulatorstep2 import Simulator2
from simulatorstep3 import Simulator3
from simulatorstep4 import Simulator4
from simulatorfirstclass import SimulatorFirst
from simulatorsecondclass import SimulatorSecond
from simulatorthirdclass import SimulatorThird

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

# Runs clairvoyant algorithm
arms = [0, 0, 0, 0, 0]
sim = Simulator2(DIRICHLET, PRICE_MATRIX, arms, 200, DIST, SECONDARY_PRODUCTS_PROB1, LAMBDA)
print(clairvoyant(sim, 300))  # If run with the second parameter = 1, can be used in the step 5
