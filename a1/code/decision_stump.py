import numpy as np
import utils
from sklearn.preprocessing import normalize

class DecisionStumpEquality:

    def __init__(self):
        pass

    def fit(self, X, y):
        N, D = X.shape # shape = (no. of arrays, no. of elements in array)
        # first column of the vector X contains the longitude and the second element contains the latitude
        #D =2
        # print(X.shape)
        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y,minlength=D)    
        '''
        np.bincount(np.array([0, 1, 1, 1, 2, 3, 7]))
        array([1, 3, 1, 1, 0, 0, 0, 1])
        4,5,6 all 0s
        '''
        
        # Get the index of the largest value in count.  
        # Thus, y_mode is the mode (most popular value) of y, either 1 or 0
        y_mode = np.argmax(count) 

        self.splitSat = y_mode
        self.splitNot = None
        self.splitVariable = None
        self.splitValue = None

        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        minError = np.sum(y != y_mode) #sum elements of array where it is not y mode

        # Loop over features looking for the best split
        X = np.round(X) # round up

        for d in range(D): # first longitude then latitude
            for n in range(N):
                # Choose value to equate to
                value = X[n, d] # for each longitude and latitude

                # Find most likely class(1 or 0) for each split(the split here is equal or not to the value)
                y_sat = utils.mode(y[X[:,d] == value]) #X[:,d] means keep column d
                # so for each longitude see how many 1 or 0 and then take the most common
                y_not = utils.mode(y[X[:,d] != value]) 

                # Make predictions
                y_pred = y_sat * np.ones(N) #0 or 1 * array of ones
                y_pred[X[:, d] != value] = y_not # replace index of cities not at longitude with the value red or blue
                # print(y_pred)
                # places where the city does not match longitude find the index and insert if it is red or blue
                # X[:, d] != value is true or false
                # Compute error
                errors = np.sum(y_pred != y)

                # Compare to minimum error so far
                if errors < minError:
                    # This is the lowest error, store this value so 35 is the lowest value
                    # find the best split
                    minError = errors
                    self.splitVariable = d
                    self.splitValue = value
                    self.splitSat = y_sat
                    self.splitNot = y_not

    def predict(self, X):

        M, D = X.shape
        X = np.round(X)

        if self.splitVariable is None:
            return self.splitSat * np.ones(M)

        yhat = np.zeros(M)

        for m in range(M):
            if X[m, self.splitVariable] == self.splitValue:
                yhat[m] = self.splitSat
            else:
                yhat[m] = self.splitNot

        return yhat


'''
Create a DecisionStumpErrorRateclass 
to do this, and report the updated error you obtain by using 
inequalities instead of discretizing and testing equality
'''


class DecisionStumpErrorRate:

    def __init__(self):
        pass

    def fit(self, X, y):
        N, D = X.shape # shape = (no. of arrays, no. of elements in array)
        # first column of the vector X contains the longitude and the second element contains the latitude
        #D =2
        # print(X.shape)
        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y,minlength=1)    
        '''
        np.bincount(np.array([0, 1, 1, 1, 2, 3, 7]))
        array([1, 3, 1, 1, 0, 0, 0, 1])
        4,5,6 all 0s
        in this case it is array([20,30])
        '''
        # Get the index of the largest value in count.  
        # Thus, y_mode is the mode (most popular value) of y, either 1 or 0
        y_mode = np.argmax(count) 

        self.splitSat = y_mode
        self.splitNot = None
        self.splitVariable = None
        self.splitValue = None

        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        minError = np.sum(y != y_mode) #sum elements of array where it is not y mode

        # Loop over features looking for the best split
        X = np.round(X) # round up
        for d in range(D): # first longitude then latitude
            for n in range(N):
                # Choose value to equate to
                value = X[n, d] # for each longitude and latitude, this is threshhold?

                # Find most likely class(1 or 0) for each split(the split here is equal or not to the value)
                y_sat = utils.mode(y[X[:,d] > value]) #X[:,d] means keep column d
                # so for each longitude see how many 1 or 0 and then take the most common
                y_not = utils.mode(y[X[:,d] <= value]) 

                # Make predictions
                y_pred = y_sat * np.ones(N) #0 or 1 * array of ones
                y_pred[X[:, d] <= value] = y_not # replace index of cities not at longitude with the value red or blue
                # print(y_pred)
                # places where the city does not match longitude find the index and insert if it is red or blue
                # X[:, d] != value is true or false
                # Compute error
                errors = np.sum(y_pred != y)

                # Compare to minimum error so far
                if errors < minError:
                    # This is the lowest error, store this value so 35 is the lowest value
                    # find the best split
                    minError = errors
                    self.splitVariable = d
                    self.splitValue = value
                    self.splitSat = y_sat
                    self.splitNot = y_not

    def predict(self, X):

        M, D = X.shape

        if self.splitVariable is None:
            return self.splitSat * np.ones(M)

        yhat = np.zeros(M)

        for m in range(M):
            if X[m, self.splitVariable] > self.splitValue:
                yhat[m] = self.splitSat
            else:
                yhat[m] = self.splitNot

        return yhat



"""
A helper function that computes the entropy of the 
discrete distribution p (stored in a 1D numpy array).
The elements of p should add up to 1.
This function ensures lim p-->0 of p log(p) = 0
which is mathematically true (you can show this with l'Hopital's rule), 
but numerically results in NaN because log(0) returns -Inf.
"""
def entropy(p): # entropy of labels: proportion of examples that satisfy threshold (1/10, 9/10)
    plogp = 0*p # initialize full of zeros
    plogp[p>0] = p[p>0]*np.log(p[p>0]) # only do the computation when p>0
    return -np.sum(plogp)

class DecisionStumpInfoGain(DecisionStumpErrorRate ):
    def fit(self , X, y, splitFeatures=None):
        N, D = X.shape #D is longitude and latitude
        #Address the trivial case where we do not split
        count = np.bincount(y) 
        #Compute total entropy (needed for information gain)
        p = count/np.sum(count) #Convert counts to probabilities e.g. 20 zeroes/total
        entropyTotal = entropy(p)

        maxGain = 0
        self.splitVariable = None
        self.splitValue = None
        self.splitSat = np.argmax(count)
        self.splitNot = None 
        #Check if labels are not all equal
        if np.unique(y).size<=1:
            return 
        if splitFeatures is None :
            splitFeatures = range(D)
        for d in splitFeatures :
            thresholds = np.unique(X[: ,d]) # unique gives number of elements which are unique
            for value in thresholds [: -1]:
                #Count number of class labels where the feature is greater than threshold
                yvals = y[X[: ,d] > value] 
                # print (X[: ,d] > value) #array of true and false
                count1 = np.bincount(yvals , minlength=len(count))
                count0 = count-count1
                #Compute infogain
                p1 = count1/np.sum(count1)
                p0 = count0/np.sum(count0)
                H1 = entropy(p1)
                H0 = entropy(p0)
                prob1 = np.sum(X[:,d] > value)/N
                prob0 = 1 - prob1
                infoGain = entropyTotal-prob1*H1-prob0*H0
                # assert infoGain >= 0
                # Compare to minimum error so far
                if infoGain>maxGain :#This is the highest information gain, store this value
                    maxGain = infoGain
                    splitVariable = d
                    splitValue = value
                    splitSat = np.argmax(count1)
                    splitNot = np.argmax(count0)

        self.splitVariable = splitVariable
        self.splitValue = splitValue
        self.splitSat = splitSat
        self.splitNot = splitNot

    def predict(self, X):
        return super().predict(X)