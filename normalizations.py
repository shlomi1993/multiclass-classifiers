# Written by Shlomi Ben-Shushan


import numpy as np


def minmax(data):
    """
    Implementation of Min-Max normalization method.

    :param data: a list of raw floats.
    :return: list of min-max normalized floats.
    """
    normalized = np.zeros(data.shape)
    for i in range(len(data[0])):
        v = data[:, i]
        minimum = min(v)
        maximum = max(v)
        if minimum != maximum:
            normalized[:, i] = (v - minimum) / (maximum - minimum)
    return normalized


def zscore(data):
    """
    Implementation of Z-Score normalization method.

    :param data: a list of raw floats
    :return: list of z-score normalized floats.
    """
    normalized = np.zeros(data.shape)
    for i in range(len(data[0])):
        v = data[:, i]
        mean = np.mean(v)
        std_dev = v.std()
        if std_dev != 0:
            normalized[:, i] = (v - mean) / std_dev
    return normalized
