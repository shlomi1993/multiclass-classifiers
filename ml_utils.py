# Written by Shlomi Ben-Shushan


import numpy as np


def shuffle(objects, labels):
    """
    This functions gets examples (objects and correspondent labels), shuffle
    them, and return shuffled lists of objects and correspondent labels.

    :param objects: a list of objects (x values).
    :param labels:  a list of labels (y values).
    :return: same objects and labels but in a shuffled order
    """
    if len(objects) != len(labels):
        raise Exception("Number of objects doesn't match the number of labels")
    examples = list(zip(objects, labels))
    np.random.shuffle(examples)
    shuffled_objects, shuffled_labels = zip(*examples)
    return np.asarray(shuffled_objects), np.asarray(shuffled_labels)


def create_w(objects, labels):
    """
    This function gets objects and labels and returns a zero-matrix from the
    order of the number of classes times number of examples.

    :param objects: a list of objects (x values).
    :param labels:  a list of labels (y values).
    :return: zero-matrix w.
    """
    return np.zeros((len(list(set(labels))), len(objects[0])), dtype=float)


def predict(w, test_x):
    """
    This function is the test-phase of Perceptron, SVM and PA algorithms.
    It gets a model and test-objects, calculate y_hats and returns test-labels.

    :param w: model matrix - the product of Perceptron, SVM or PA.
    :param test_x: the x values of the test-set.
    :return: predictions of test_y - the y values of the test-set.
    """
    test_y = []
    for i, example in enumerate(test_x):
        y_hat = np.argmax(np.dot(w, example))
        test_y.append(y_hat)
    return test_y
