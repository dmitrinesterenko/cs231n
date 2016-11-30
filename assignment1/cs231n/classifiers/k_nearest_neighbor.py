import numpy as np

class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  __DEBUG__ = False

  def __init__(self):
    pass

  def debug(self, args):
    if self.__DEBUG__:
      print args

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
     y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y

  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
     of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    #X = X.reshape(num_test, 32*32*3)

    self.debug(X.shape)
    self.debug(X[0])
    self.debug("Training shape %s, %s" % (self.X_train.shape))
    self.debug("First training row %s" % (self.X_train[0]))
    #assert(self.X_train.all() == X.all())
    #X_train = self.X_train.reshape(num_train, 32, 32, 3)
    for i in xrange(num_test):
      for j in xrange(num_train):
        dists[i,j] = np.sqrt(np.sum((X[i] - self.X_train[j])**2))
    #####################################################################
    # TODO:                                                             #
    # Compute the l2 distance between the ith test point and the jth    #
    # training point, and store the result in dists[i, j]. You should   #
    # not use a loop over dimension.                                    #
    #####################################################################
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    self.debug("distances shape %s, %s" % (dists.shape))
    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
    #######################################################################
    # TODO:                                                               #
    # Compute the l2 distance between the ith test point and all training #
    # points, and store the result in dists[i, :].                        #
    #######################################################################
    # go through the shape of dists and what each dists i.j contains
    # after each of the row computations
      dists[i,:] = np.sqrt(np.sum((X[i] - self.X_train)**2,axis=1))
    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    #dists = np.sqrt(np.sum((X - np.transpose(self.X_train))**2))
    #import pdb; pdb.set_trace()

    #dists = np.sqrt(np.sum(X**2 - X*np.transpose(self.X_train) + self.X_train**2))
    #import pdb; pdb.set_trace()
    #dists = np.sqrt((np.sum(X**2, axis=1) - np.sum(X*np.transpose(self.X_train), axis=1) + np.sum(self.X_train**2, axis=1)))

    # use (a-b)^2 = a^2 - 2ab + b^2
    X_squared = np.sum(X**2, axis=1)
    X_train_squared = np.sum(self.X_train**2, axis=1)
    projected_X_squared = X_squared[:,np.newaxis]

    self.debug("projected_X_squared.shape %s, %s" % (projected_X_squared.shape))
    dists = np.sqrt(projected_X_squared+X_train_squared - 2*np.dot(X,self.X_train.T))


    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    self.debug("dists.shape %s,%s" % (dists.shape))
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].
    """
    num_test = dists.shape[0]
    #print "dists"
    #print dists
    dists_sorted = np.argsort(dists)
    #print dists_sorted.shape
    y_pred = np.zeros(num_test)
    #for i in xrange(num_test):
    for i in xrange(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # testing point, and use self.y_train to find the labels of these       #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################
      # these are K indexes of the closest points to the ith point
      # these ought to be indexes to the closest images
      # because dists_sorted ought to be just the indexes
      # that represent the values of the dists after they have been sorted
      closest = dists_sorted[i][0:k]

      closest_y = [self.y_train[x] for x in closest]

      #self.pdebug("dists_sorted[i] %s closest %s closest_y %s" % (dists_sorted[i], closest, closest_y))
      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################
      # label appearance frequency in sorted order
      appearances = [closest_y.count(j) for j in sorted(closest_y)]
      # max times that a label appears
      max_appearance = max(appearances)
      # grab the first label with the most appearances
      y_pred[i] = [m for m in sorted(closest_y) if closest_y.count(m) == max_appearance][0]
      self.debug("y_pred[%s] %s" %(i, y_pred[i]))
      #########################################################################
      #                           END OF YOUR CODE                            #
      #########################################################################
    return y_pred

