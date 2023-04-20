# -*- coding: utf-8 -*-
"""
Messy functions of Neural Network
"""
import csv
import random
import numpy as np


# Dataset
def load_dataset(path, includes_header=True):
    """
    Turn .csv into list of data
    1st row is header
    Last column in output (int stating with 0)
    :param path: csv file path
    :param includes_header: Does the csv file have header?
    :return: List without the header
    """
    file = open(path, 'r')
    dataset = list(csv.reader(file, delimiter=','))
    file.close()

    if includes_header:
        return dataset[1:]  # Without the header
    else:
        return dataset


def heart_split_dataset(dataset, train=0.8, shuffle=False):
    """
    Split dataset into train and test datasets
    :param dataset: Your dataset
    :param train: How much should be used as train data (0 - 1)
    :param shuffle: Change the order of data in dataset
    :return: (train data list, test data list)
    """
    if shuffle:
        random.shuffle(dataset)

    groups = set()
    for idx in range(len(dataset)):
        dataset[idx] = [float(num) for num in dataset[idx]]
        dataset[idx][-1] = int(dataset[idx][-1])
        groups.add(dataset[idx][-1])

    count = [0 for _ in range(len(groups))]
    for row in dataset:
        count[row[-1]] += 1

    train_data, test_data = [], []
    tmp_count = [0 for _ in range(len(groups))]
    for row in dataset:
        if tmp_count[row[-1]] < int(count[row[-1]] * train):
            tmp_count[row[-1]] += 1
            train_data.append(row)
        else:
            test_data.append(row)

    return train_data, test_data


def cancer_split_dataset(dataset, train=0.8, shuffle=False):
    """
    Split dataset into train and test datasets
    :param dataset: Your dataset
    :param train: How much should be used as train data (0 - 1)
    :param shuffle: Change the order of data in dataset
    :return: (train data list, test data list)
    """
    if shuffle:
        random.shuffle(dataset)

    groups = set()
    for idx in range(len(dataset)):
        dataset[idx] = dataset[idx][1:]
        dataset[idx][0] = 0 if dataset[idx][0] == 'B' else 1
        dataset[idx] = [float(num) for num in dataset[idx]]
        dataset[idx][0] = int(dataset[idx][0])
        groups.add(dataset[idx][0])

    count = [0 for _ in range(len(groups))]
    for row in dataset:
        count[row[0]] += 1

    train_data, test_data = [], []
    tmp_count = [0 for _ in range(len(groups))]
    for row in dataset:
        if tmp_count[row[0]] < int(count[row[0]] * train):
            tmp_count[row[0]] += 1
            train_data.append(row)
        else:
            test_data.append(row)

    return train_data, test_data


def xy_split_dataset(dataset, y_matrix=True, data='heart'):
    """
    Turn the dataset into X (input list) and y (output list)
    :param dataset: Your dataset
    :param y_matrix: Return y as metrix (True by Default)
    :param data: What type of dataset are we using (heart by default)
    :return: (np.array(X), np.array(y))
    """
    X, y = [], []

    if data == 'heart':
        if y_matrix:
            for row in dataset:
                X.append(row[:-1])
                if row[-1] == 0:
                    y.append(np.array([1, 0]))
                else:
                    y.append(np.array([0, 1]))
        else:
            for row in dataset:
                X.append(row[:-1])
                y.append([row[-1]])

    else:
        if y_matrix:
            for row in dataset:
                X.append(row[1:])
                if row[0] == 0:
                    y.append(np.array([1, 0]))
                else:
                    y.append(np.array([0, 1]))
        else:
            for row in dataset:
                X.append(row[1:])
                y.append([row[0]])

    return np.array(X), np.array(y)


# Activation Functions
def sigmoid(x):
    """ Sigmoid Activation Function """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """ Sigmoid Activation Function Derivation """
    return x * (1 - x)


def relu(x):
    """ ReLU Activation Function """
    return np.maximum(0, x)


def relu_derivative(x):
    """ Sigmoid Activation Function Derivation """
    return np.where(x <= 0, 0, 1)
