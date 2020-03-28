
# basics
import argparse
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

# our code
import linear_model
import utils

url_amazon = "https://www.amazon.com/dp/%s"

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)
    io_args = parser.parse_args()
    question = io_args.question

    if question == "1":

        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(f,names=("user","item","rating","timestamp"))

        print("Number of ratings:", len(ratings))
        print("The average rating:", np.mean(ratings["rating"]))

        n = len(set(ratings["user"]))
        d = len(set(ratings["item"]))
        print("Number of users:", n)
        print("Number of items:", d)
        print("Fraction nonzero:", len(ratings)/(n*d))

        X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind = utils.create_user_item_matrix(ratings)
        print(type(X))
        print("Dimensions of X:", X.shape)

    elif question == "1.1":
        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(f,names=("user","item","rating","timestamp"))
        X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind = utils.create_user_item_matrix(ratings)
        X_binary = X != 0
        
        # YOUR CODE HERE FOR Q1.1
        item_total_stars = X.sum(axis=0)
        top_ratings_index = item_total_stars.argmax()
        print("The top rated item:", item_inverse_mapper[top_ratings_index])
        print("The number of stars of top rated item:",item_total_stars.max())

        # YOUR CODE HERE FOR Q1.1.2
        user_most_reviews_index_matrix = X.nonzero() #for each item a user has reviewed, the users index will appear once in this matrix
        user_most_reviews_index_row = np.asarray(user_most_reviews_index_matrix[0])
        user_number_reviews = np.bincount(user_most_reviews_index_row) #counting frequency of user index to see who reviewed the most items
        user_most_reviews_index = user_most_reviews_index_row[user_number_reviews.argmax()]
        user_most_reviews = user_inverse_mapper[user_most_reviews_index]
        print("The user with most reviews is",user_most_reviews)
        user_most_reviews_total = user_number_reviews.max()
        print("The total reviews for this user is",user_most_reviews_total)

        # YOUR CODE HERE FOR Q1.1.3
        print(user_number_reviews)
        plt.yscale('log', nonposy='clip')
        plt.hist(user_number_reviews)
        plt.xlabel('Number of ratings')
        plt.ylabel('Number of users')
        plt.title('num_ratings_per_user')
        fig1 = plt.gcf()
        plt.show()
        fig1.savefig('../figs/num_ratings_per_user.png')

        item_num_reviews = np.bincount(np.asarray(user_most_reviews_index_matrix[1]))
        plt.yscale('log', nonposy='clip')
        plt.hist(item_num_reviews)
        plt.xlabel('Number of ratings')
        plt.ylabel('Number of items')
        plt.title('num_ratings_per_item')
        fig2 = plt.gcf()
        plt.show()
        fig2.savefig('../figs/num_ratings_per_item.png')

        plt.xlabel('Number of stars')
        plt.ylabel('Number of ratings')
        plt.title('Ratings')
        plt.hist(ratings["rating"]) #histogram of every rating given
        fig3 = plt.gcf()
        plt.show()
        fig3.savefig('../figs/ratings.png')


        


    elif question == "1.2":
        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(f,names=("user","item","rating","timestamp"))
        X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind = utils.create_user_item_matrix(ratings)
        X_binary = X != 0

        grill_brush = "B00CFM0P7Y"
        grill_brush_ind = item_mapper[grill_brush]
        grill_brush_vec = X[:,grill_brush_ind] #select column of grill brush

        print(url_amazon % grill_brush)

        # YOUR CODE HERE FOR Q1.2
        #tuning matrix
        d = len(set(ratings["item"]))
        to_keep = list(set(range(d))-{grill_brush_ind})    
        X_wo_query = X[:,to_keep]
        new_X = X_wo_query.transpose()
        nbrs = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean').fit(new_X)
        distances, indices = nbrs.kneighbors(grill_brush_vec.transpose())
        print("Closest 5 items using Euclidean distance:")
        for i in indices[0]:
            print(item_inverse_mapper[i])
        
        print('Euclidean numer of ratings',X[:,indices[0]].getnnz(axis=0))

        normalized_X = normalize(new_X)
        nbrs = NearestNeighbors(n_neighbors=5, algorithm='brute',metric='euclidean').fit(normalized_X)
        distances, indices = nbrs.kneighbors(grill_brush_vec.transpose())
        print("Closest 5 items using normallized euclidean distance:")
        for i in indices[0]:
            print(item_inverse_mapper[i])

        nbrs = NearestNeighbors(n_neighbors=5, algorithm='brute',metric='cosine').fit(new_X)
        distances, indices = nbrs.kneighbors(grill_brush_vec.transpose())
        print("Closest 5 items using cosine distance:")
        for i in indices[0]:
            print(item_inverse_mapper[i])

        print('Cosine number of ratings:', X[:,indices[0]].getnnz(axis=0))




        # YOUR CODE HERE FOR Q1.3


    elif question == "3":
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        # Fit least-squares estimator
        model = linear_model.LeastSquares()
        model.fit(X,y)
        print(model.w)

        utils.test_and_plot(model,X,y,title="Least Squares",filename="least_squares_outliers.pdf")

    elif question == "3.1":
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        # YOUR CODE HERE

    elif question == "3.3":
        # loads the data in the form of dictionary
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        # Fit least-squares estimator
        model = linear_model.LinearModelGradient()
        model.fit(X,y)
        print(model.w)

        utils.test_and_plot(model,X,y,title="Robust (L1) Linear Regression",filename="least_squares_robust.pdf")

    elif question == "4":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        # Fit least-squares model
        model = linear_model.LeastSquares()
        model.fit(X,y)

        utils.test_and_plot(model,X,y,Xtest,ytest,title="Least Squares, no bias",filename="least_squares_no_bias.pdf")

    elif question == "4.1":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        # YOUR CODE HERE

    elif question == "4.2":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        for p in range(11):
            print("p=%d" % p)

            # YOUR CODE HERE

    else:
        print("Unknown question: %s" % question)

