import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD
    max_cls = int(y.max())
    min_cls = int(y.min())
    means = np.zeros((max_cls - min_cls + 1) * X.shape[1]).reshape(max_cls - min_cls + 1, X.shape[1])
    covmat = np.zeros(X.shape[1] * X.shape[1]).reshape(X.shape[1], X.shape[1])
    for i in range(0, max_cls - min_cls + 1) :
        select = np.where(y == float(i + 1))[0]
        means[i] = X[select, :].sum(axis = 0) / len(select)
        
    mean_of_all = X.sum(axis = 0) / X.shape[0]
    covmat_val = np.dot(np.transpose(X - mean_of_all), X - mean_of_all) / X.shape[0]
    for i in range(0, X.shape[1]) :
        covmat[i, i] = covmat_val[i, i]            
    return np.transpose(means),covmat
    
def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    max_cls = int(y.max())
    min_cls = int(y.min())
    means = np.zeros((max_cls - min_cls + 1) * X.shape[1]).reshape(max_cls - min_cls + 1, X.shape[1])
    covmats = []
    for i in range(0, max_cls - min_cls + 1) :
        covmat_dig = np.zeros(X.shape[1] * X.shape[1]).reshape(X.shape[1], X.shape[1])
        select = np.where(y == float(i + 1))[0]
        subset = X[select, :]
        means[i] = subset.sum(axis = 0) / len(select)
        covmat_val = np.dot(np.transpose(subset - means[i]), subset - means[i]) / subset.shape[0]
        for i in range(0, X.shape[1]) :
            covmat_dig[i, i] = covmat_val[i, i]
        covmats.append(covmat_dig)
        
    return np.transpose(means),covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    num_of_test_examples = Xtest.shape[0]
    ypred = np.zeros(num_of_test_examples).reshape(num_of_test_examples, 1)
    correct_num = 0
    for i in range(0, num_of_test_examples) :
        current_example = Xtest[i]
        delta = np.transpose(means) - current_example
        distance = np.dot(np.dot(delta, np.linalg.inv(covmat)), np.transpose(delta))
        top = 0
        for j in range(1, distance.shape[0]) :
            if distance[j, j] < distance[top, top] :
                top = j
        ypred[i, 0] = top + 1
        if top + 1 == int(ytest[i]) :
            correct_num += 1
    
    acc = str(float(correct_num) / num_of_test_examples * 100) + '%'          
    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    num_of_test_examples = Xtest.shape[0]
    #d = Xtest.shape[1]
    ypred = np.zeros(num_of_test_examples).reshape(num_of_test_examples, 1)
    correct_num = 0
    num_of_class = len(covmats)
    for i in range(0, num_of_test_examples) :
        current_example = Xtest[i]
        top = -1
        curr_max = 0.0
        for k in range(num_of_class) :
            mean = means[:, k]
            covmat = covmats[k]
            delta = np.transpose(mean) - current_example
            pdf = 1.0 / sqrt(np.sum(covmat ** 2)) * np.exp((-1 / 2) * np.dot(np.dot(delta, np.linalg.inv(covmat)),np.transpose(delta)))
            if pdf > curr_max :
                top = k
                curr_max = pdf
        
        ypred[i, 0] = top + 1
        if top + 1 == int(ytest[i]) :
            correct_num += 1
    
    acc = str(float(correct_num) / num_of_test_examples * 100) + '%'
    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 
	
    # IMPLEMENT THIS METHOD 
    x_inverse = np.dot(np.transpose(X), X)  
    w = np.dot(np.dot(np.linalg.pinv(x_inverse), np.transpose(X)), y)                                               
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD
    A = lambd * np.eye(X.shape[1]) + np.dot(np.transpose(X), X)
    w = np.dot(np.dot(np.linalg.pinv(A), np.transpose(X)), y)                                                 
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    # IMPLEMENT THIS METHOD
    w_col = w.reshape(w.shape[0], 1)
    y_pred = np.dot(Xtest, w_col)
    num_of_example = Xtest.shape[0]
    mse = 1.0 / num_of_example * np.sum((y_pred - ytest) ** 2)
    return mse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD
    num_of_example = X.shape[0]
    diff = y - np.dot(X, w.reshape(X.shape[1], 1)).reshape(num_of_example, 1)
    error = 1.0 / 2 * np.dot(np.transpose(diff), diff) + 1.0 / 2 * lambd * np.sum(w ** 2)
    error_grad = np.zeros((X.shape[1], 1))
    for i in range(num_of_example) :
        example = np.dot(X[i, :].reshape(1, X.shape[1]), w.reshape(X.shape[1], 1)).reshape(1, 1) - y[i]
        current_row = example * X[i, :]
        error_grad += current_row.reshape(X.shape[1], 1) 
    
    error_grad += (lambd * w.reshape(X.shape[1], 1)) 
                                                                   
    return error, error_grad.flatten()

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1)) 
	
    # IMPLEMENT THIS METHOD
    Xd = np.ones((x.shape[0], p + 1))
    for i in range(1, p + 1):
        Xd[:, i] = np.power(x, i)
    return Xd

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('/Users/weiyijiang/Desktop/ML PA2/sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('/Users/weiyijiang/Desktop/ML PA2/sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))

# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('QDA')

plt.show()
# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('/Users/weiyijiang/Desktop/ML PA2/diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('/Users/weiyijiang/Desktop/ML PA2/diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()
# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))

for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()


# Problem 5
pmax = 7
lambda_opt = 0 # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()
