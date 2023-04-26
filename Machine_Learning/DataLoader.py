# -*- coding: utf-8 -*-
"""
Dataset preparation functions
"""
import csv
import random
import numpy as np


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


def tt_split_dataset(dataset, train=0.8, data='heart', shuffle=False):
    """
    Split dataset into train and test datasets
    :param dataset: Your dataset
    :param train: How much should be used as train data (0 - 1)
    :param data: What type of dataset are we using (heart by default)
    :param shuffle: Change the order of data in dataset
    :return: (train data list, test data list)
    """
    if shuffle:
        random.shuffle(dataset)

    translate = {
        1: None,
        -1: None
    }

    groups = set()
    if data == 'heart':
        translate[-1] = 'Negative'
        translate[1] = 'Positive'
        for idx in range(len(dataset)):
            dataset[idx][-1] = (0 if dataset[idx][-1] == '0' else 1)
            dataset[idx] = [float(num) for num in dataset[idx]]
            dataset[idx][-1] = int(dataset[idx][-1])
            groups.add(dataset[idx][-1])
    elif data == 'cancer':
        translate[-1] = 'B (Negative)'
        translate[1] = 'M (Positive)'
        for idx in range(len(dataset)):
            dataset[idx].append(0 if dataset[idx][1] == 'B' else 1)
            dataset[idx] = dataset[idx][2:]
            dataset[idx] = [float(num) for num in dataset[idx]]
            dataset[idx][-1] = int(dataset[idx][-1])
            groups.add(dataset[idx][-1])
    else:
        translate[-1] = 'Negative'
        translate[1] = 'Positive'
        for idx in range(len(dataset)):
            dataset[idx][-1] = (0 if dataset[idx][-1] == '0' else 1)
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
            row[-1] = -1 if row[-1] == 0 else 1
            train_data.append(row)
        else:
            row[-1] = -1 if row[-1] == 0 else 1
            test_data.append(row)

    return train_data, test_data, translate


def xy_split_dataset(dataset):
    """
    Turn the dataset into X (input list) and y (output list)
    :param dataset: Your dataset
    :return: (np.array(X), np.array(y))
    """
    X, y = [], []

    for row in dataset:
        X.append(row[:-1])
        y.append(row[-1])

    return np.array(X), np.array(y)
