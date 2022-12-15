"""
defines the web page class

Author: Lorenzo Di Toro
"""

import numpy as np

PRODUCTS = [0, 1, 2, 3, 4]


class WebPage:
    def __init__(self, primary_product, price):
        """ Initializes the web page, picks randomly the products to show in the secondary slots
        :param primary_product: integer indicating the primary product
        :param price: selling price for the primary product
        """
        self.primary_product = primary_product
        self.price = price
        self.first_slot, self.second_slot = np.random.choice(PRODUCTS, 2, replace=False)
