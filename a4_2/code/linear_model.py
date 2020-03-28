import numpy as np
from numpy.linalg import solve
import findMin
from scipy.optimize import approx_fprime
import utils

class logReg:
    # Logistic Regression
    def __init__(self, verbose=0, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self,X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMin(self.funObj, self.w,
                                      self.maxEvals, X, y, verbose=self.verbose)
    def predict(self, X):
        return np.sign(X@self.w)



class logRegL0(logReg):
    # L0 Regularized Logistic Regression
    def __init__(self, L0_lambda=1.0, verbose=2, maxEvals=400):
        self.verbose = verbose
        self.L0_lambda = L0_lambda
        self.maxEvals = maxEvals

    def fit(self, X, y):
        n, d = X.shape
        minimize = lambda ind: findMin.findMin(self.funObj,
                                                  np.zeros(len(ind)),
                                                  self.maxEvals,
                                                  X[:, ind], y, verbose=0)
        selected = set()
        selected.add(0)
        minLoss = np.inf
        oldLoss = 0
        bestFeature = -1

        while minLoss != oldLoss:
            oldLoss = minLoss
            print("Epoch %d " % len(selected))
            print("Selected feature: %d" % (bestFeature))
            print("Min Loss: %.3f\n" % minLoss)

            for i in range(d):
                if i in selected:
                    continue

                selected_new = selected | {i} # tentatively add feature "i" to the seected set

                # TODO for Q2.3: Fit the model with 'i' added to the features,
                # then compute the loss and update the minLoss/bestFeature


            selected.add(bestFeature)

        self.w = np.zeros(d)
        self.w[list(selected)], _ = minimize(list(selected))


class leastSquaresClassifier:
    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((self.n_classes,d))

        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y==i] = 1
            ytmp[y!=i] = -1

            # solve the normal equations
            # with a bit of regularization for numerical reasons
            self.W[i] = np.linalg.solve(X.T@X+0.0001*np.eye(d), X.T@ytmp)

    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)
    
class logLinearClassifier:
    def __init__(self, maxEvals=500, verbose=1):
        self.maxEvals = maxEvals
        self.verbose = verbose
        
    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.w = np.zeros((self.n_classes,d))
        for c in range(self.n_classes):
            # set correct class to 1 and rest to -1
            yc = -np.ones(len(y))
            yc[y==c] = 1
            self.w[c], f = findMin.findMin(self.funObj, self.w[c], self.maxEvals,
                                              X, yc, verbose=self.verbose)
    
    def funObj(self, w, X, y):
        ywx = np.multiply(y,X@w.T)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-ywx)))

        # Calculate the gradient value
        g = (-y / (1. + np.exp(ywx))).T @ X
        
        return f, g
            
    def predict(self, X):
        return np.argmax(X@self.w.T, axis=1)
    
class softmaxClassifier:
    def __init__(self, maxEvals=500, verbose=1):
        self.maxEvals = maxEvals
        self.verbose = verbose
        
    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.w = np.zeros(self.n_classes * d)
        self.w, f = findMin.findMin(self.funObj, self.w, self.maxEvals,
                                    X, y, verbose=self.verbose)
    
    def funObj(self, w, X, y):
        n, d = X.shape
        W = np.reshape(w, (self.n_classes, d))

        # Calculate the function value
        wx = np.sum(np.multiply(W[y],X),axis=1)
        f = np.sum(-wx + np.log(np.sum(np.exp(X @ W.T), axis=1)))

        # Calculate the gradient value
        g = np.zeros((self.n_classes,d))
        for c in range(self.n_classes):
            I = np.zeros(n)
            I[y==c] = 1
            
            p_num = np.exp(X @ W[c].T)
            p_den = np.sum(np.exp(X @ W.T), axis=1)
            p = np.divide(p_num, p_den)
            
            pI = np.repeat((p-I)[:, np.newaxis], d, axis=1)
            g[c] = np.sum(np.multiply(X, pI), axis=0)
        
        g = g.flatten()
        return f, g
            
    def predict(self, X):
        n, d = X.shape
        W = np.reshape(self.w, (self.n_classes, d))
        return np.argmax(X@W.T, axis=1)