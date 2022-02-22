# Written by Shlomi Ben-Shushan


import sys
from normalizations import zscore, minmax
from classifiers import *


def count_hits(predictions, test_y):
    count = 0
    for i in range(len(test_y)):
        if predictions[i] == test_y[i]:
            count += 1
    return count


def create_cross_validation_sets(objects, labels):
    train_x, train_y, test_x, test_y = [], [], [], []
    chunk_size = int(0.2 * len(objects))
    objects_chunks = [objects[i:i + chunk_size] for i in range(len(objects))[::chunk_size]]
    labels_chunks = [labels[i:i + chunk_size] for i in range(len(labels))[::chunk_size]]
    for i in range(5):
        a = i
        b = (i + 1) % 5
        c = (i + 2) % 5
        d = (i + 3) % 5
        e = (i + 4) % 5
        train_x.append(np.concatenate([objects_chunks[a], objects_chunks[b], objects_chunks[c], objects_chunks[d]], axis=0))
        train_y.append(np.concatenate([labels_chunks[a], labels_chunks[b], labels_chunks[c], labels_chunks[d]], axis=0))
        test_x.append(objects_chunks[e])
        test_y.append(labels_chunks[e])
    return train_x, train_y, test_x, test_y


def cross_validation(objects, labels, algo, params):
    train_x, train_y, test_x, test_y = create_cross_validation_sets(objects, labels)
    ratios = []
    for i in range(5):
        if len(params) == 1:
            predictions = algo(train_x[i], train_y[i], params[0], test_x[i])
        elif len(params) == 2:
            predictions = algo(train_x[i], train_y[i], params[0], params[1], test_x[i])
        elif len(params) == 3:
            predictions = algo(train_x[i], train_y[i], params[0], params[1], params[2], test_x[i])
        else:
            raise "Wrong number of params."
        hits = count_hits(predictions, test_y[i])
        ratios.append(round(hits / len(predictions), 4))
    return ratios, round(np.average(ratios), 4)


def calibrate_knn(objects, labels):
    current_max = 0
    best = None
    for k in range(2, int(len(objects) / 2)):
        results, average = cross_validation(objects, labels, knn, [k])
        if current_max < average:
            current_max = average
            best = (k, average)
            # print("Current best: k=" + str(k) + "=> accuracy=" + str(average))
    return best


def calibrate_perceptron(objects, labels):
    current_max = 0
    best = None
    for epochs in [e for e in range(10, 160, 10)]:
        eta = round(0.1, 4)
        while eta <= 1.0:
            results, average = cross_validation(objects, labels, perceptron, [epochs, eta])
            if current_max < average:
                current_max = average
                best = (epochs, eta, average)
                # print("Current best: epochs=" + str(epochs) + " eta=" + str(eta) + "=> accuracy=" + str(average))
            eta = round(eta + 0.1, 4)
    return best


def calibrate_svm(objects, labels):
    current_max = 0
    best = None
    for epochs in [e for e in range(10, 160, 10)]:
        eta = round(0.1, 4)
        while eta <= 1.0:
            lamb = round(0.1, 4)
            while lamb <= 1.0:
                results, average = cross_validation(objects, labels, svm, [epochs, eta, lamb])
                if current_max < average:
                    current_max = average
                    best = (epochs, eta, lamb, average)
                    # print("Current best: epochs=" + str(epochs) + " eta=" + str(eta) + " lambda=" + str(lamb) + "=> accuracy=" + str(average))
                lamb = round(lamb + 0.1, 4)
            eta = round(eta + 0.1, 4)
    return best


def calibrate_pa(objects, labels):
    current_max = 0
    best = None
    for epochs in [e for e in range(10, 1010, 10)]:
        results, average = cross_validation(objects, labels, passive_aggressive, [epochs])
        if current_max < average:
            current_max = average
            best = (epochs, average)
            # print("Current best: epochs=" + str(epochs) + "=> accuracy=" + str(average))
    return best


def find_params(objects, labels, title):
    print(title)
    a = calibrate_knn(objects, labels)
    print(" * KNN: K=" + str(a[0]) + " => accuracy=" + str(a[1]))
    b = calibrate_perceptron(objects, labels)
    print(" * Perceptron: epochs=" + str(b[0]) + " eta=" + str(b[1]) + " => accuracy=" + str(b[2]))
    c = calibrate_svm(objects, labels)
    print(" * SVM: epochs=" + str(c[0]) + " eta=" + str(c[1]) + " lambda=" + str(c[2]) + " => accuracy=" + str(c[3]))
    d = calibrate_pa(objects, labels)
    print(" * PA: epochs=" + str(d[0]) + " => accuracy=" + str(d[1]))
    print()


def calibrate():
    objects, labels = shuffle(np.genfromtxt(sys.argv[1], delimiter=","), np.genfromtxt(sys.argv[2], dtype=int))
    print("Running tests...\n")
    find_params(objects, labels, "Best params without normalization:")
    find_params(zscore(objects), labels, "Best params with Z-Score normalization:")
    find_params(minmax(objects), labels, "Best params with Min-Max normalization:")
    print("Done!\n")


calibrate()
