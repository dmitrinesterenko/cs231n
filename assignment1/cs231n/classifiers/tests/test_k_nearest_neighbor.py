import nose
import numpy as np
from cs231n.classifiers import KNearestNeighbor
from cs231n.data_utils import load_CIFAR10

cifar10_dir = '../datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

num_training = 10
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]
num_test = 1
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

# make rows out of each of the images
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print "X train"
print X_train.shape, X_test.shape, y_train.shape
print X_train


# Create a kNN classifier instance.
# Remember that training a kNN classifier is a noop:
# the Classifier simply remembers the data and does no further processing

classifier = KNearestNeighbor()
classifier.train(X_train, y_train)
dists = []

def test_distance_when_train_and_test_are_the_same():
  dists = classifier.compute_distances_two_loops(X_train)
  print "dists"
  print dists
  assert(dists.all() == np.zeros(dists.shape).all())

def test_predict_labels():
  dists = classifier.compute_distances_two_loops(X_train)
  classifier.predict_labels(dists,1)

def test_distance_with_one_loop():
  dists = classifier.compute_distances_two_loops(X_train)

  dists_one = classifier.compute_distances_one_loop(X_train)
  difference = np.linalg.norm(dists - dists_one, ord='fro')
  assert(difference < 0.01)

def test_distance_with_no_loops():
  dists = classifier.compute_distances_two_loops(X_train)

  dists_two = classifier.compute_distances_no_loops(X_train)
  difference = np.linalg.norm(dists - dists_two, ord='fro')
  assert(difference < 0.01)
