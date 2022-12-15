"""
the clairvoyant algorithm. Finds best arms configuration in a Bruteforce fashion

Authors: Lorenzo Di Toro
"""

import numpy as np
import itertools
from simulatorstep3 import Simulator3


def clairvoyant(simulation, repetition):
    """ Runs a number of time the simulation for each arm configuration and gets the one with the best average
    :param simulation: a simulation object
    :param repetition: number of repetitions of simulations for each arms configurations
    :return: bet arm configuration
    """
    i = 0
    candidates = [0, 1, 2, 3]
    # Finds all possible arm configurations
    configurations = list(itertools.product(candidates, repeat=5))

    # Parameters
    old_reward = np.zeros((5, 4))
    old_reward = old_reward.sum()
    best_configuration = []

    # For each configuration runs a number of times the simulation
    for i in range(len(configurations)):
        new_reward = np.zeros((5, 4))
        j = 0
        for j in range(repetition):
            simulation.arms_configuration = configurations[i]
            new_reward += simulation.run()
        # Calculates the avg reward
        new_reward = new_reward.sum() / repetition
        if new_reward > old_reward:
            best_configuration = list(configurations[i]).copy()
            old_reward = new_reward
    return best_configuration, old_reward
