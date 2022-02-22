# Written by Shlomi Ben-Shushan


from ml_utils import *


def knn(train_x, train_y, test_x, k):
    """
    This function is my implementation of KNN algorithm.
    Logic: for every test-object, find the k train-objects closest to it and
    determine its test-label as the label most common among the train-labels
    correspondent to the train-objects.

    :param train_x: x values of the train-set.
    :param train_y: y values of the train-set.
    :param test_x: x values of the test-set.
    :param k: KNN's hyper-parameter - the number of neighbors.
    :return: predictions of test_y which is the y values of the test-set.
    """
    test_y = []
    classes = np.array(list(set(train_y)))
    for test_object in test_x:
        distances = []
        for train_object, train_label in zip(train_x, train_y):
            distance = np.linalg.norm(test_object - train_object)
            distances.append([distance, train_label])
        distances = sorted(distances, key=lambda index: index[0])
        counters = [0] * len(classes)
        for i in range(k):
            for j in range(len(classes)):
                if distances[i][1] == classes[j]:
                    counters[j] += 1
        test_y.append(np.argmax(counters))
    return test_y


def perceptron(train_x, train_y, test_x, epochs, eta):
    """
    This function is my implementation of Perceptron algorithm.
    Logic: for "epochs" iterations, calculate y_hat, and if it is not equals to
    the real y, then the algo missed so update it according to the instructions,
    and divide eta by 2 to reduce the learning rate.

    :param train_x: x values of the train-set.
    :param train_y: y values of the train-set.
    :param test_x: x values of the test-set.
    :param epochs: Perceptron's hyper-parameter - the number of iterations.
    :param eta: Perceptron's hyper-parameter - the learning rate.
    :return: predictions of test_y which is the y values of the test-set.
    """
    w = create_w(train_x, train_y)
    for epoch in range(epochs):
        for x, y in zip(train_x, train_y):
            y_hat = np.argmax(np.dot(w, x))
            if y_hat != y:
                w[y] = w[y] + (eta * x)
                w[y_hat] = w[y_hat] - (eta * x)
                eta = round(eta / 2, 4)
    return predict(w, test_x)


def svm(train_x, train_y, test_x, epochs, eta, lamb):
    """
    This function is my implementation of SVM algorithm.
    Logic: for "epochs" iterations, calculate y_hat, and if it is not equals to
    the real y, then the algo missed so update it according to the instructions,
    and divide eta and lamb(da) by 2 to reduce the learning rate and the regulation.

    :param train_x: x values of the train-set.
    :param train_y: y values of the train-set.
    :param test_x: x values of the test-set.
    :param epochs: SVM's hyper-parameter - the number of iterations.
    :param eta: SVM's hyper-parameter - the learning rate.
    :param lamb: (short for lambda) SVM's hyper-parameter - the regulation.
    :return: predictions of test_y which is the y values of the test-set.
    """
    w = create_w(train_x, train_y)
    for epoch in range(epochs):
        for x, y in zip(train_x, train_y):
            y_hat = np.argmax(np.dot(w, x))
            if y_hat != y:
                c = 1 - eta * lamb
                w[y] = c * w[y] + eta * x
                w[y_hat] = c * w[y_hat] - eta * x
                for v in range(w.shape[0]):
                    if v != y and v != y_hat:
                        w[v:] = c * w[v:]
                eta = round(eta / 2, 4)
                lamb = round(lamb / 2, 4)
    return predict(w, test_x)


def passive_aggressive(train_x, train_y, test_x, epochs):
    """
    This function is my implementation of PA algorithm.
    Logic: for "epochs" iterations, calculate y_hat, and if it is not equals to
    the real y, then the algo missed so update it according to the instructions,
    including the calculation and using of tau.

    :param train_x: x values of the train-set.
    :param train_y: y values of the train-set.
    :param test_x: x values of the test-set.
    :param epochs: SVM's hyper-parameter - the number of iterations.
    :return: predictions of test_y which is the y values of the test-set.
    """
    w = create_w(train_x, train_y)
    for epoch in range(epochs):
        for x, y in zip(train_x, train_y):
            y_hat = np.argmax(np.dot(w, x))
            if y != y_hat:
                x_norm = np.linalg.norm(x)
                if x_norm != 0:
                    loss = max(0, 1 - np.dot(w[y], x) + np.dot(w[y_hat], x))
                    tau = loss / (2 * (x_norm ** 2))
                    w[y] += tau * x
                    w[y_hat] -= tau * x
    return predict(w, test_x)
