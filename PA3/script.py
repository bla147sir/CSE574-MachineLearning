import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
import pickle
from sklearn.svm import SVC

def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    x = np.hstack((np.ones((n_data, 1)), train_data))
    w = initialWeights.reshape(n_features + 1, 1)
    y = labeli

    theta = sigmoid(np.dot(x, w))
    error_temp = y * np.log(theta) + (1.0 - y) * np.log(1.0 - theta)
    error = float(-1/n_data) * (np.sum(error_temp))
    error_grad = np.sum((theta - y) * x, axis=0) * float(1 / n_data)

    return error, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    x = np.hstack((np.ones((data.shape[0], 1)), data))
    probability = sigmoid(np.dot(x, W))      # compute the posterior probability P(y = Ck|x)
    label = np.argmax(probability, axis=1)     # assign x to class Ck that maximizes P(y = Ck|x)
    label = label.reshape(data.shape[0], 1)

    return label


def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    initialWeights = params
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    n_class = labeli.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    x = np.hstack((np.ones((n_data, 1)), train_data))
    w = initialWeights.reshape(n_feature + 1, n_class)
    y = labeli

    temp = np.zeros((n_data, 1))
    for i in range(n_class):
        temp += np.exp(np.dot(x, w[:,[i]]))

    probability = np.zeros((n_data, n_class))
    for i in range(n_class) :
        probability[:, [i]] = np.divide(np.exp(np.dot(x, w[:, [i]])), temp)

  #  for k in range(n_data):
  #      for i in range(n_class):
  #          error += y[k, [i]] * np.log(probability[k, [i]])

    error = np.sum(y * np.log(probability))
    error = np.sum(error) * (-1)
    print('error = '+ str(error))

    for i in range(n_class):
        error_grad[:, [i]] = np.dot(x.T, probability[:, [i]] - y[:, [i]]).reshape(n_feature + 1, 1)

    error_grad = error_grad.flatten()
    #print('error_gra = '+str(error_grad.shape))

    return error, error_grad

def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    n_data = data.shape[0]
    n_class = W.shape[1]    # n_class = 10
    label = np.zeros((n_data, 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    x = np.hstack((np.ones((n_data, 1)), data))
    temp = np.zeros((n_data, 1))
    for i in range(n_class):
        temp += np.exp(np.dot(x, W[:, [i]]))

    probability = np.zeros((n_data, n_class))
    for i in range(n_class):
        probability[:, [i]] = np.divide(np.exp(np.dot(x, W[:, [i]])), temp)

    label = np.argmax(probability, axis=1)  # assign x to class Ck that maximizes P(y = Ck|x)
    label = label.reshape(n_data, 1)

    return label


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

f1 = open('params.pickle', 'wb')
pickle.dump(W, f1)
f1.close()

"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################

print('\n1. linear kernel')
clf = SVC(kernel = 'linear')
clf.fit(train_data, train_label.flatten())
print('\n Training set Accuracy:' + str(100 * clf.score(train_data, train_label)) + '%')
print('\n Validation set Accuracy:' + str(100 * clf.score(validation_data, validation_label)) + '%')
print('\n Testing set Accuracy:' + str(100 * clf.score(test_data, test_label)) + '%')

print('\n2. radial basis, gamma = 1')
clf = SVC(kernel = 'rbf', gamma = 1.0)
clf.fit(train_data, train_label.flatten())
print('\n Training set Accuracy:' + str(100 * clf.score(train_data, train_label)) + '%')
print('\n Validation set Accuracy:' + str(100 * clf.score(validation_data, validation_label)) + '%')
print('\n Testing set Accuracy:' + str(100 * clf.score(test_data, test_label)) + '%')

print('\n2. radial basis, gamma = 0')
clf = SVC(kernel = 'rbf')
clf.fit(train_data, train_label.flatten())
print('\n Training set Accuracy:' + str(100 * clf.score(train_data, train_label)) + '%')
print('\n Validation set Accuracy:' + str(100 * clf.score(validation_data, validation_label)) + '%')
print('\n Testing set Accuracy:' + str(100 * clf.score(test_data, test_label)) + '%')

print('\n2. radial basis, different C')
train_result = np.zeros(11)
valid_result = np.zeros(11)
test_result = np.zeros(11)
C = np.zeros(11)
C[0] = 1.0
C[1] = 10.0
C[2] = 20.0
C[3] = 30.0
C[4] = 40.0
C[5] = 50.0
C[6] = 60.0
C[7] = 70.0
C[8] = 80.0
C[9] = 90.0
C[10] = 100.0

for i in range(11):
    clf = SVC(C = C[i],kernel = 'rbf')
    clf.fit(train_data, train_label.flatten())
    train_result[i] = clf.score(train_data, train_label)
    valid_result[i] = clf.score(validation_data, validation_label)
    test_result[i] = clf.score(test_data, test_label)

# Save results in rbf_diff_C.pickle
pickle.dump((train_result, valid_result, test_result),open("rbf_diff_C.pickle","wb"))

"""
Script for Extra Credit Part
"""
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}   #100

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')

f2 = open('params_bonus.pickle', 'wb')
pickle.dump(W_b, f2)
f2.close()