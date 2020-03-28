import numpy as np

class NaiveBayes:
    # Naive Bayes implementation.
    # Assumes the feature are binary.
    # Also assumes the labels go from 0,1,...C-1, so labels are not binary

    def __init__(self, num_classes, beta=0):
        self.num_classes = num_classes
        self.beta = beta

    def fit(self, X, y):
        N, D = X.shape # D is number of features

        # Compute the number of class labels
        C = self.num_classes

        # Compute the probability of each class i.e p(y==c)
        counts = np.bincount(y)
        p_y = counts / N

        # Compute the conditional probabilities i.e.
        # p(x(i,j)=1 | y(i)==c) as p_xy is actually proportion of examples with feature x(i,j) =1 when label is c
        # p(x(i,j)=0 | y(i)==c) as p_xy
        # Assume independent features and for each feature insert conditional prob ie proportion for that class label
        prop = np.ones((D,C))
        for i in range(C):
            id_label = np.where(y==i)[0]
            for j in range(D):
                feature_given_label = X[id_label][:,j]
                n_feature = np.count_nonzero(feature_given_label)
                proportion = n_feature/len(feature_given_label)
                prop[j,i]=proportion
        p_xy = prop
        self.p_y = p_y
        self.p_xy = p_xy



    def predict(self, X):

        N, D = X.shape
        C = self.num_classes
        p_xy = self.p_xy
        p_y = self.p_y

        y_pred = np.zeros(N)
        for n in range(N):

            probs = p_y.copy() # initialize with the p(y) terms
            for d in range(D):
                if X[n, d] != 0:
                    probs *= p_xy[d, :]
                else:
                    probs *= (1-p_xy[d, :])

            y_pred[n] = np.argmax(probs)

        return y_pred
