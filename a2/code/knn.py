"""
Implementation of k-nearest neighbours classifier
"""

import operator

import numpy as np
from scipy import stats
import utils

class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X # just memorize the trianing data
        self.y = y 

    def predict(self, Xtest):
        # Locate the most similar neighbors
        # for each training point how far the other 2000 test pts are away from it
        dist = utils.euclidean_dist_squared(self.X, Xtest) # (400,2000)
        index_sorted_dist = np.argsort(dist,axis=0) 
        # returns array of indexes of sorted array [213,512,100] returns [2,0,1] so point number 2 is closest to tr pt
        # return k nearest neighbours for each test pt

        neighbours = index_sorted_dist[: self.k] # get first k rows of x axis test pt y axis isindices of tr points

        # Make a classification prediction with neighbors
        classes = self.y[neighbours]
        # Each column ie test pt has all the tr pt classes 
        majority_label = stats.mode(classes)
        prediction = majority_label[0][0]
        return prediction

    def getError(self,prediction,ytest):
        print("Error is: {}".format(np.sum(prediction != ytest)/len(ytest)))
        # print("Training accuracy is: {}".format(np.sum(prediction != self.y)/len(self.y)))
