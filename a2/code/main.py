# basics
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np


# sklearn imports
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# our code
import utils

from knn import KNN

from naive_bayes import NaiveBayes

from decision_stump import DecisionStumpErrorRate, DecisionStumpEquality, DecisionStumpInfoGain
from decision_tree import DecisionTree
from random_tree import RandomTree
# from random_forest import RandomForest

from kmeans import Kmeans
from sklearn.cluster import DBSCAN

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question


    if question == "1":
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]        
        model = DecisionTreeClassifier(max_depth=2, criterion='entropy', random_state=1)
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print("Training error: %.3f" % tr_error)
        print("Testing error: %.3f" % te_error)

    elif question == "1.1":
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]

        depths = np.arange(1,16) # depths to try
        tr_error = np.zeros(depths.size)
        te_error = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTree(max_depth=max_depth)
            model.fit(X, y)

            y_pred = model.predict(X)
            tr_error[i] = np.mean(y_pred != y)

            y_pred = model.predict(X_test)
            te_error[i] = np.mean(y_pred != y_test)
            print(i)
        plt.plot(depths,tr_error,'r--',depths,te_error,label="errorrate")
        plt.show()

    elif question == '1.2':
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        print(type(dataset))
        print(type(dataset["X"]))
        X, y = dataset["X"], dataset["y"]
        print(X.shape)

        X_tr, X_val = np.array_split (X,2)
        y_tr, y_val = np.array_split (y,2)
        print(X_tr.shape)
        depths = np.arange(1,15) # depths to try
        tr_error = np.zeros(depths.size)
        tv_error = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTree(max_depth=max_depth)
            model.fit(X_tr, y_tr)

            y_pred = model.predict(X_tr)
            tr_error[i] = np.mean(y_pred != y_tr)

            y_pred = model.predict(X_val)
            tv_error[i] = np.mean(y_pred != y_val)
            print(i)
        plt.plot(depths,tr_error,'r--',depths,tv_error,label="errorrate")
        plt.show()



    elif question == '2.2':
        dataset = load_dataset("newsgroups.pkl")

        X = dataset["X"]
        y = dataset["y"]
        print(X.shape)
        print(y.shape)
        X_valid = dataset["Xvalidate"]
        y_valid = dataset["yvalidate"]
        groupnames = dataset["groupnames"]
        wordlist = dataset["wordlist"]
        print(wordlist[50])
        for i in np.nonzero(X[500]):
            print(wordlist[i])

        print(groupnames[y[500]])




    elif question == '2.3':
        dataset = load_dataset("newsgroups.pkl")

        X = dataset["X"]
        y = dataset["y"]
        X_valid = dataset["Xvalidate"]
        y_valid = dataset["yvalidate"]

        print("d = %d" % X.shape[1])
        print("n = %d" % X.shape[0])
        print("t = %d" % X_valid.shape[0])
        print("Num classes = %d" % len(np.unique(y)))

        model = NaiveBayes(num_classes=4)
        model.fit(X, y)
        y_pred = model.predict(X_valid)
        v_error = np.mean(y_pred != y_valid)
        print("Naive Bayes (ours) validation error: %.3f" % v_error)

        clf = BernoulliNB()
        clf.fit(X, y)
        print("Naive Bayes (sklearn) validation error: %.3f" % (1-clf.score(X,y)))

    

    elif question == '3':
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset['X']
        y = dataset['y']
        Xtest = dataset['Xtest']
        ytest = dataset['ytest']

        for i in [1]:
            knn = KNN(i)
            knn.fit(X,y)
            print("Training")
            tr_prediction = knn.predict(X)
            knn.getError(tr_prediction,y)
            print("Testing")
            prediction = knn.predict(Xtest)
            knn.getError(prediction, ytest)

        utils.plotClassifier(knn,X,y)
        neigh = KNeighborsClassifier(n_neighbors=1)
        neigh.fit(X, y)
        # print(neigh.predict(ytest))
        print("Sklearn KNN: {}".format(1-neigh.score(X,y)))
        utils.plotClassifier(neigh,X,y)



    elif question == '4':
        dataset = load_dataset('vowel.pkl')
        X = dataset['X']
        y = dataset['y']
        X_test = dataset['Xtest']
        y_test = dataset['ytest']
        print("\nn = %d, d = %d\n" % X.shape)

        def evaluate_model(model):
            model.fit(X,y)

            y_pred = model.predict(X)
            tr_error = np.mean(y_pred != y)

            y_pred = model.predict(X_test)
            te_error = np.mean(y_pred != y_test)
            print("    Training error: %.3f" % tr_error)
            print("    Testing error: %.3f" % te_error)

        print("Decision tree info gain")
        evaluate_model(DecisionTree(max_depth=np.inf, stump_class=DecisionStumpInfoGain))



    elif question == '5':
        X = load_dataset('clusterData.pkl')['X']

        model = Kmeans(k=4)
        model.fit(X)
        y = model.predict(X)
        plt.scatter(X[:,0], X[:,1], c=y, cmap="jet")

        fname = os.path.join("..", "figs", "kmeans_basic.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    elif question == '5.1':
        X = load_dataset('clusterData.pkl')['X']



    elif question == '5.2':
        X = load_dataset('clusterData.pkl')['X']



    elif question == '5.3':
        X = load_dataset('clusterData2.pkl')['X']

        model = DBSCAN(eps=1, min_samples=3)
        y = model.fit_predict(X)

        print("Labels (-1 is unassigned):", np.unique(model.labels_))
        
        plt.scatter(X[:,0], X[:,1], c=y, cmap="jet", s=5)
        fname = os.path.join("..", "figs", "density.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
        
        plt.xlim(-25,25)
        plt.ylim(-15,30)
        fname = os.path.join("..", "figs", "density2.png")
        plt.savefig(fname)
        print("Figure saved as '%s'" % fname)
        
    else:
        print("Unknown question: %s" % question)
