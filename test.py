# Written by Shlomi Ben-Shushan


import sys
from normalizations import zscore
from classifiers import *


# Get training-set and shuffle it.
train_x = np.genfromtxt(sys.argv[1], delimiter=",")
train_y = np.genfromtxt(sys.argv[2], dtype=np.int64)
test_x = np.genfromtxt(sys.argv[3], delimiter=",")
train_x, train_y = shuffle(train_x, train_y)

# Normalize the data for Perceptron, SVM and PA.
train_x_z = zscore(train_x)
test_x_z = zscore(test_x)

# Run algorithms...
knn_test_y = knn(train_x, train_y, test_x, k=5)
perceptron_test_y = perceptron(train_x_z, train_y, test_x_z, epochs=90, eta=0.2)
svm_test_y = svm(train_x_z, train_y, test_x_z, epochs=80, eta=0.1, lamb=0.8)
pa_test_y = passive_aggressive(train_x_z, train_y, test_x_z, epochs=190)

# Output results in the requested format.
open(sys.argv[4], "w").close()
f = open(sys.argv[4], "a")
for knn_yhat, perceptron_yhat, svm_yhat, pa_yhat in zip(knn_test_y, perceptron_test_y, svm_test_y, pa_test_y):
    f.write(f"knn: {knn_yhat}, perceptron: {perceptron_yhat}, svm: {svm_yhat}, pa: {pa_yhat}\n")
f.close()
