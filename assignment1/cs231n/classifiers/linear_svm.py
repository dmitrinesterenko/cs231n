import matplotlib.pyplot as plt
import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  previous_margin = 0.0
  num_incorrect  = 0
  num_nonzero = 0
  num_skipped = 0
  loss_matrix = np.zeros((num_train, num_classes))
  score_matrix = np.zeros((num_train,1))
  #import pdb; pdb.set_trace()

  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    score_matrix[i] = correct_class_score
    num_incorrect = 0
    for j in xrange(num_classes):
      if j == y[i]:
        num_skipped += 1
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      loss_matrix[i,j] = np.maximum(0, margin)
      if margin > 0:
        num_nonzero += 1
        loss += margin
        num_incorrect += 1
        # analytic gradient goes here
        # when Wj
        dW[:,j] += X[i]
        #if j == 0:
          #print dW[:,j]
          #import pdb; pdb.set_trace()
      #dW[i,j] = (margin - previous_margin)*W[i,j]
      #previous_margin = margin
    # when Wyi
    dW[:,y[i]] += - num_incorrect * X[i]
    fig = plt.figure()
    ax = fig.gca()
    ax.matshow(dW, cmap=plt.cm.autumn_r) #_r => reverse the standard color map
    plt.show()


  # dW is over all the training examples, we want to get an average and so
  # divide by their number.
  dW /= num_train

  # Add regularization gradient to the gradient
  dW += 0.5 * reg * 2 * W

  print "Non zero num skipped ", num_nonzero, num_skipped
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW, loss_matrix, score_matrix


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
    #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = np.dot(X,W)
  max_comparison = np.zeros(scores.shape)
  loss_matrix = np.zeros(scores.shape)
  normalized_scores = np.zeros(scores.shape)
  non_zero_scores = np.zeros(scores.shape)
  correct_scores = np.zeros(scores.shape[0])

  # Want to subtract from each training row to "correct class score"
  # which corresponds to for the ith row to scores[i,y[i]]
  # so from each entry in scores[i] - scores[i,y[i]]
  correct_scores = scores[np.arange(scores.shape[0]),y]
  loss_matrix = np.maximum(max_comparison, scores - correct_scores[:,np.newaxis] + 1)
  print " Non-zero before resetting y ", np.count_nonzero(loss_matrix)
  # Want something like this to happen, the 0th value of y is 9, set the value
  # of position 9 in row 0 to 0
  # y[0] = 9
  # (Pdb) loss_matrix[0,y[0]] = 0
  # (Pdb) loss_matrix[0]
  #  array([ 1.71950797,  0.93701105,  0.77294409,  1.89414119,  0.67763448,
  #      0.        ,  1.8138858 ,  2.07053251,  1.64546695,  0.        ])
  # Use integer array indexing
  # https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
  loss_matrix[np.arange(loss_matrix.shape[0]),y] = 0
  # count how many elements as a result are non-zero to compare with how many
  # elements were > than 0 in the for loop implementation
  count_deviants = np.count_nonzero(loss_matrix)
  print " Non-zero ", count_deviants

  loss = np.sum(loss_matrix) / scores.shape[0]
  loss += 0.5 * reg * np.sum(W * W)
  #import pdb; pdb.set_trace()

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  #if (score - correct_score > 1)
  # when Wj
  #non_zero_scores = scores[np.where(scores > 0)]
  #mask = ma.masked_array(loss_matrix, loss_matrix > 0)


  loss_matrix[loss_matrix > 0] = 1
  dW = np.dot(X.T, loss_matrix) / 500
  print "dW.shape %d, %d" % dW.shape

  #TODO
  # when Wyi
  # 1. Need a pythonic way to tell for each row how many things are equal to one from the
  # loss_matrix (this is the count_wrong_vector
  # 2. Need to apply to each of the dW[yi] = - count_wrong_vector * X

  #  (Pdb) y[i]
  #  2
  #  (Pdb) dW[:,2].shape
  #  (3073,)
  #  (Pdb) num_incorrect
  #  9
  #  (Pdb) X[i].shape
  #  (3073,)   import pdb; pdb.set_trace()
  # import pdb; pdb.set_trace()
  dW[np.arange(dW.shape[0]),y] = - np.dot(X.T, np.sum(loss_matrix, axis=0))

  dW += 0.5 * reg * 2 * W

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW, loss_matrix, correct_scores
