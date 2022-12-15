"""
defines the class User

Author: Lorenzo Di Toro
"""
import random

import numpy as np


class User:
    def __init__(self, reservation_prices):
        """ initializes an object user with a given array of prices. the prices are the maximum value given from the
        user to that item
        :param reservation_prices: array of 5 prices for the 5 items
        """
        self.reservation_prices = reservation_prices
        self.primary_history = []
        self.cart = []

    def buy(self, webpage, number_elements):
        """ given the selling price of the item, decides if the users buys it or not
        :param webpage: webpage where the user is
        :param number_elements: how many unity of the product to buy eventually. if null than it's random
        :return: True if it does buy the item, False otherwise
        """
        # Checks if the quantity of elements to buy is fixed or not
        if number_elements is None:
            number_elements = np.random.choice(100, 1)

        if webpage.price <= self.reservation_prices[webpage.primary_product]:
            self.cart.append([number_elements, webpage.primary_product])
            return True
        else:
            return False

    def visit_other_pages(self, webpage, matrix, lambda_):
        """ This method is called only if the users buys the primary product of the webpage. It creates an event,
        and if this event and based on that it returns 0, 1 or 2 products, whose pages have to be visited
        :param webpage: the webpage obj that represent the actual webpage
        :param matrix: the matrix containing all the probability for the visiting of the secondary slots
        :param lambda_: lambda
        :return: array of integer representing the webpage to create and visit
        """
        # Initializes the return variable
        webpages_to_visit = []

        # Gets needed data from the webpage
        first_product = webpage.first_slot
        second_product = webpage.second_slot
        primary = webpage.primary_product
        self.primary_history.append(primary)

        # Calculates the probability to visit the secondary webpage products
        prob_visit_first = matrix[primary][first_product]
        prob_visit_second = matrix[primary][second_product] * lambda_
        if first_product in self.primary_history:
            prob_visit_first = 0
        if second_product in self.primary_history:
            prob_visit_second = 0

        # Creates a random variable between 0 and 1
        event = random.random()
        if event < prob_visit_second:
            webpages_to_visit.append(second_product)
        if event < prob_visit_first:
            webpages_to_visit.append(first_product)

        return webpages_to_visit
