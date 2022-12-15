"""
Simulator for step 7

Authors: Lorenzo Di Toro
"""
import random

import numpy as np
from user import User
from webpage import WebPage


class SimulatorFirst:
    def __init__(self, dirichlet, price_matrix, arms_configuration, number_users, distributions, slot_matrix, lambda_):
        """ Initializes the class simulation and shows the price list and the slot matrix. Alpha ratios are
        initialized here to, not passed as a parameter, as well as all the users.
        :param dirichlet: fixed dirichlet distribution
        :param price_matrix: all the possible prices for all the different items
        :param arms_configuration: arms to pull chosen by some algorithm
        :param number_users: fixed number of users per day
        :param distributions: matrix of matrices of the different distributions for the different type of users
        :param slot_matrix: matrix representing the graph
        :param lambda_: lambda
        """
        self.price_matrix = price_matrix
        self.slot_matrix = slot_matrix
        self.lambda_ = lambda_
        self.distributions = distributions
        self.dirichlet = dirichlet
        self.arms_configuration = arms_configuration
        self.number_users = number_users

        # Initializes the group of users
        count = 0
        self.users = []
        for count in range(number_users):
            # Users class randomly picked
            type_ = 0  # Classes fixed in this case
            i = 0
            res_values = []
            for i in range(5):
                res_values.append(np.random.choice(self.price_matrix[i], p=self.distributions[type_][i]))
            usr = User(res_values)
            self.users.append(usr)

    def run(self):
        """ creates as many webpages as "useful" ones (the not competitors ones) and check weather the user assigned to it
        buys the primary object. After that checks if the users clicks on any of the slots and creates the right web
        pages
        :return: 5x4 matrix with total reward per price for this group of users
        """
        # Initializes all the not concurrent web pages
        count = 0
        webpages = []
        useful_users = []
        inter0 = self.dirichlet[0]
        inter1 = inter0 + self.dirichlet[1]
        inter2 = inter1 + self.dirichlet[2]
        inter3 = inter2 + self.dirichlet[3]
        inter4 = inter3 + self.dirichlet[4]
        inter5 = inter4 + self.dirichlet[5]
        for count in range(len(self.users)):
            rand = random.random()
            if inter0 < rand <= inter1:
                web = WebPage(0, self.price_matrix[0][self.arms_configuration[0]])
                webpages.append(web)
                useful_users.append(self.users[count])
            elif inter1 < rand <= inter2:
                web = WebPage(1, self.price_matrix[1][self.arms_configuration[1]])
                webpages.append(web)
                useful_users.append(self.users[count])
            elif inter2 < rand <= inter3:
                web = WebPage(2, self.price_matrix[2][self.arms_configuration[2]])
                webpages.append(web)
                useful_users.append(self.users[count])
            elif inter3 < rand <= inter4:
                web = WebPage(3, self.price_matrix[3][self.arms_configuration[3]])
                webpages.append(web)
                useful_users.append(self.users[count])
            elif inter4 < rand <= inter5:
                web = WebPage(4, self.price_matrix[4][self.arms_configuration[4]])
                webpages.append(web)
                useful_users.append(self.users[count])

        # Initializes the matrix to return
        reward_matrix = np.zeros((5, 4))

        # Check if each user buys the product of the page where it lands and creates the eventual webpages
        count = 0
        for count in range(len(webpages)):
            if useful_users[count].buy(webpages[count], random.randint(1, 3)):  # Random number of items bought
                i = 0
                pages = useful_users[count].visit_other_pages(webpages[count], self.slot_matrix, self.lambda_)
                # adds the reward to the reward matrix multiplying it by the number of items bought
                reward_matrix[webpages[count].primary_product][
                    self.arms_configuration[webpages[count].primary_product]] += \
                    self.price_matrix[webpages[count].primary_product][
                        self.arms_configuration[webpages[count].primary_product]] * \
                    useful_users[count].cart[len(useful_users[count].cart) - 1][0]
                if pages is not None:
                    for i in range(len(pages)):
                        web = WebPage(pages[i], self.price_matrix[pages[i]][self.arms_configuration[pages[i]]])
                        webpages.append(web)
                        useful_users.append(useful_users[count])
        return reward_matrix

    def monte_carlo_sampling(self, number_runs):
        """ runs monte carlo sampling algorithm. Assumes that the user always buys the primary product. the algorithm
        surely finishes, same consideration as for the navigation of the user between the webpages. It can't click on a
        webpage showing a primary product already bought
        :return: 5x5 matrix of probabilities
        """
        probability_matrix = np.zeros((5, 5))

        # For each item runs the given number of simulations
        i = 0
        for i in range(5):
            clients = []
            webpages = []
            j = 0
            # Creates a given number of users for each item
            for j in range(number_runs):
                # Creates new users
                type_ = j % 3
                k = 0
                res_values = []
                for k in range(5):
                    res_values.append(np.random.choice(self.price_matrix[i], p=self.distributions[type_][k]))
                usr = User(res_values)
                clients.append(usr)

                # Creates web pages with the correct primary item
                webpages.append(WebPage(i, self.price_matrix[i][0]))  # The price is not important

            j = 0
            for j in range(len(clients)):
                t = 0
                # Assumes that the user buys the page
                pages = clients[j].visit_other_pages(webpages[j], self.slot_matrix, self.lambda_)
                if pages is not None:
                    for t in range(len(pages)):
                        probability_matrix[i][pages[t]] += 1
                        # Adds a client and the corresponding webpage to simulate the eventual chain
                        k = 0
                        type_ = 0
                        res_values = []
                        webpages.append(WebPage(pages[t], self.price_matrix[i][0]))  # The price is not important
                        for k in range(5):
                            res_values.append(np.random.choice(self.price_matrix[i], p=self.distributions[type_][k]))
                        clients.append(User(res_values))
            probability_matrix[i] = probability_matrix[i] / len(clients)
        return probability_matrix

