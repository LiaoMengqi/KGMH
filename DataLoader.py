import os
import numpy as np


class DataLoader(object):
    def __init__(self, dataset, path):
        """
        param dataset: the name of dataset
        param path: the path of dataset
        """
        self.path = path
        self.dataset = dataset

        # data
        self.train = None
        self.valid = None
        self.test = None

    def load_data(self):
        """
        Initialize training set, validation set and test set.
        """
        with open(self.path + self.dataset + './train.txt') as file:
            train = []
            for line in file:
                line = str(line)
                line = line.strip()
                items = line.split('\t')
                train.append([int(items[0]), int(items[1]), int(items[2])])
