import numpy as np
from numpy.linalg import solve
import findMin
from scipy.optimize import approx_fprime
import utils

def log_1_plus_exp_safe(x):
    # compute log(1+exp(x)) in a numerically safe way, avoiding overflow/underflow issues
    out = np.log(1+np.exp(x))
    out[x > 100] = x[x>100]
    out[x < -100] = np.exp(x[x < -100])
    return out

class logRegL2():
    # L2 Regularized Logistic Regression (no intercept)
    def __init__(self, lammy=1.0, verbose=0, maxEvals=100):
        self.verbose = verbose
        self.lammy = lammy
        self.maxEvals = maxEvals

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        # f = np.sum(np.log(1. + np.exp(-yXw)))
        f = np.sum(log_1_plus_exp_safe(-yXw))

        # Add L2 regularization
        f += 0.5 * self.lammy * np.sum(w**2)

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res) + self.lammy * w

        return f, g

    def fit(self,X, y):
        n, d = X.shape

        # Initial guess
        utils.check_gradient(self, X, y, d, verbose=self.verbose)
        self.w, f = findMin.findMin(self.funObj, np.zeros(d), self.maxEvals, X, y, verbose=self.verbose)

    def predict(self, X):
        return np.sign(X@self.w)

def kernel_RBF(X1, X2, sigma=0.5):
    K = np.zeros((np.size(X1,0),np.size(X2,0)))
    for i in range(np.size(X1,0)):
        for j in range(np.size(X2,0)):
            K[i][j]=np.exp(-np.linalg.norm((X1[i]-X2[j]),ord=2)**2/(2*(sigma**2)))
    return K

def kernel_poly(X1, X2, p=2):
    return ((1+X1@X2.T)**p) 

def kernel_linear(X1, X2):
    return X1@X2.T# X is (300,2) produces nxn (300x300)matrix of similarities between rows

class kernelLogRegL2():
    def __init__(self, lammy=0.01, verbose=0, maxEvals=100, kernel_fun=kernel_RBF, **kernel_args): #good use of polymerism in kernel_fun
        self.verbose = verbose
        self.lammy = lammy
        self.maxEvals = maxEvals
        self.kernel_fun = kernel_fun
        self.kernel_args = kernel_args

    def funObj(self, u, K, y):
        yKu = y * (K@u) # error here K is empty

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yKu)))
        f = np.sum(log_1_plus_exp_safe(-yKu))

        # Add L2 regularization
        f += 0.5 * self.lammy * u.T@K@u

        # Calculate the gradient value
        res = - y / (1. + np.exp(yKu))
        g = (K.T@res) + self.lammy * K@u

        return f, g

    def fit(self, X, y):
        n, d = X.shape
        self.X = X

        K = self.kernel_fun(X,X, **self.kernel_args) #X-X is 0, so K is 0, why is it XX here, somehow it works for

        utils.check_gradient(self, K, y, n, verbose=self.verbose)
        self.u, f = findMin.findMin(self.funObj, np.zeros(n), self.maxEvals, K, y, verbose=self.verbose)

    def predict(self, Xtest):
        Ktest = self.kernel_fun(Xtest, self.X, **self.kernel_args)
        return np.sign(Ktest@self.u)




