# -*- coding: utf-8 -*-

import os
import sys
import random
import math
import numpy as np
import heapq
from evaluate import Evaluator
from surprise import AlgoBase, PredictionImpossible, NormalPredictor, SVD, SVDpp, KNNBasic
from surprise.model_selection import GridSearchCV
from movielens import MovieLens


def main():
    np.random.seed(0)
    random.seed(0)

    ml = MovieLens()
    print("Loading movie ratings...")
    evaluationData = ml.loadMovieLensLatestSmall()
    print("\nComputing movie popularity ranks so we can measure novelty later...")
    rankings = ml.getPopularityRanks()

    print("Running ContentKNN:")
    # Construct an Evaluator to, you know, evaluate them
    evaluator = Evaluator(evaluationData, rankings)
    contentKNN = ContentKNNAlgorithm()
    evaluator.AddAlgorithm(contentKNN, "ContentKNN")

    print("Running User KNN:")
    UserKNN = KNNBasic(sim_options = {'name': 'cosine', 'user_based': True})
    evaluator.AddAlgorithm(UserKNN, "User KNN")

    print("Running Item KNN:")
    ItemKNN = KNNBasic(sim_options = {'name': 'cosine', 'user_based': False})
    evaluator.AddAlgorithm(ItemKNN, "Item KNN")


    print("Searching for best parameters...")
    param_grid = {'n_epochs': [20, 30], 'lr_all': [0.005, 0.010], 'n_factors': [50, 100]}
    gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
    gs.fit(evaluationData)
    print("Best RMSE score attained: ", gs.best_score['rmse'])
    print(gs.best_params['rmse'])
    params = gs.best_params['rmse']
    SVDtuned = SVD(n_epochs = params['n_epochs'], lr_all = params['lr_all'], n_factors = params['n_factors'])
    evaluator.AddAlgorithm(SVDtuned, "SVD")

    print("Running Random predictor as a sanity check")
    Random = NormalPredictor()
    evaluator.AddAlgorithm(Random, "Random")

    # Compare
    evaluator.Evaluate(False)
    evaluator.SampleTopNRecs(ml)


class ContentKNNAlgorithm(AlgoBase):
    def __init__(self, k=40, sim_options={}):
        AlgoBase.__init__(self)
        self.k = k

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        # Compute item similarity matrix based on content attributes
        # Load up genre vectors for every movie
        ml = MovieLens()
        genres = ml.getGenres()
        years = ml.getYears()
        mes = ml.getMiseEnScene()
        print("Computing content-based similarity matrix...")
        # Compute genre distance for every movie combination as a 2x2 matrix
        self.similarities = np.zeros((self.trainset.n_items, self.trainset.n_items))
        for thisRating in range(self.trainset.n_items):
            if (thisRating % 100 == 0):
                print(thisRating, " of ", self.trainset.n_items)
            for otherRating in range(thisRating+1, self.trainset.n_items):
                thisMovieID = int(self.trainset.to_raw_iid(thisRating))
                otherMovieID = int(self.trainset.to_raw_iid(otherRating))
                genreSimilarity = self.computeGenreSimilarity(thisMovieID, otherMovieID, genres)
                yearSimilarity = self.computeYearSimilarity(thisMovieID, otherMovieID, years)
                mesSimilarity = self.computeMiseEnSceneSimilarity(thisMovieID, otherMovieID, mes)
                self.similarities[thisRating, otherRating] = genreSimilarity * yearSimilarity * mesSimilarity
                self.similarities[otherRating, thisRating] = self.similarities[thisRating, otherRating]
        print("...done.")
        return self

    def computeGenreSimilarity(self, movie1, movie2, genres):
        genres1 = genres[movie1]
        genres2 = genres[movie2]
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(genres1)):
            x = genres1[i]
            y = genres2[i]
            sumxx += x * x
            sumyy += y * y
            sumxy += x * y

        return sumxy/math.sqrt(sumxx*sumyy)

    def computeYearSimilarity(self, movie1, movie2, years):
        diff = abs(years[movie1] - years[movie2])
        sim = math.exp(-diff / 10.0)
        return sim

    def computeMiseEnSceneSimilarity(self, movie1, movie2, mes):
        mes1 = mes[movie1]
        mes2 = mes[movie2]
        if (mes1 and mes2):
            shotLengthDiff = math.fabs(mes1[0] - mes2[0])
            colorVarianceDiff = math.fabs(mes1[1] - mes2[1])
            motionDiff = math.fabs(mes1[3] - mes2[3])
            lightingDiff = math.fabs(mes1[5] - mes2[5])
            numShotsDiff = math.fabs(mes1[6] - mes2[6])
            return shotLengthDiff * colorVarianceDiff * motionDiff * lightingDiff * numShotsDiff
        else:
            return 0

    def estimate(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')
        # Build up similarity scores between this item and everything the user rated
        neighbors = []
        for rating in self.trainset.ur[u]:
            genreSimilarity = self.similarities[i,rating[0]]
            neighbors.append( (genreSimilarity, rating[1]) )
        # Extract the top-K most-similar ratings
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])
        # Compute average sim score of K neighbors weighted by user ratings
        simTotal = weightedSum = 0
        for (simScore, rating) in k_neighbors:
            if (simScore > 0):
                simTotal += simScore
                weightedSum += simScore * rating
        if (simTotal == 0):
            raise PredictionImpossible('No neighbors')
        predictedRating = weightedSum / simTotal
        return predictedRating


if __name__ == '__main__':
    main()

