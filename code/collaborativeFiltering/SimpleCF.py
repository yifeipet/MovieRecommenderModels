# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))

from MovieLens import MovieLens
from surprise import KNNBasic
import heapq
from collections import defaultdict
from operator import itemgetter

class SimpleCF:
    def __init__(self, testSubject='85', k=10):
        self.testSubject = testSubject
        self.k = k

        self.ml = MovieLens()
        data = self.ml.loadMovieLensLatestSmall()
        self.trainSet = data.build_full_trainset()

        print("User-User:")
        self.user()
        print('\nItem-Item:')
        self.item()


    def user(self):
        sim_options = {'name': 'cosine', 'user_based': True}
        model = KNNBasic(sim_options=sim_options)
        model.fit(self.trainSet)
        simsMatrix = model.compute_similarities()

        # Get top N similar users to our test subject
        # (Alternate approach would be to select users up to some similarity threshold - try it!)
        testUserInnerID = self.trainSet.to_inner_uid(self.testSubject)
        similarityRow = simsMatrix[testUserInnerID]

        similarUsers = []
        for innerID, score in enumerate(similarityRow):
            if (innerID != testUserInnerID):
                similarUsers.append( (innerID, score) )

        kNeighbors = heapq.nlargest(self.k, similarUsers, key=lambda t: t[1])

        # Get the stuff they rated, and add up ratings for each item, weighted by user similarity
        candidates = defaultdict(float)
        for similarUser in kNeighbors:
            innerID = similarUser[0]
            userSimilarityScore = similarUser[1]
            theirRatings = self.trainSet.ur[innerID]
            for rating in theirRatings:
                candidates[rating[0]] += (rating[1] / 5.0) * userSimilarityScore

        # Build a dictionary of stuff the user has already seen
        watched = {}
        for itemID, rating in self.trainSet.ur[testUserInnerID]:
            watched[itemID] = 1

        # Get top-rated items from similar users:
        pos = 0
        for itemID, ratingSum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
            if not itemID in watched:
                movieID = self.trainSet.to_raw_iid(itemID)
                print(self.ml.getMovieName(int(movieID)), ratingSum)
                pos += 1
                if (pos > 10):
                    break


    def item(self):
        sim_options = {'name': 'cosine', 'user_based': False}

        model = KNNBasic(sim_options=sim_options)
        model.fit(self.trainSet)
        simsMatrix = model.compute_similarities()
        testUserInnerID = self.trainSet.to_inner_uid(self.testSubject)

        # Get the top K items we rated
        testUserRatings = self.trainSet.ur[testUserInnerID]
        kNeighbors = heapq.nlargest(self.k, testUserRatings, key=lambda t: t[1])

        # Get similar items to stuff we liked (weighted by rating)
        candidates = defaultdict(float)
        for itemID, rating in kNeighbors:
            similarityRow = simsMatrix[itemID]
            for innerID, score in enumerate(similarityRow):
                candidates[innerID] += score * (rating / 5.0)

        # Build a dictionary of stuff the user has already seen
        watched = {}
        for itemID, rating in self.trainSet.ur[testUserInnerID]:
            watched[itemID] = 1

        # Get top-rated items from similar users:
        pos = 0
        for itemID, ratingSum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
            if not itemID in watched:
                movieID = self.trainSet.to_raw_iid(itemID)
                print(self.ml.getMovieName(int(movieID)), ratingSum)
                pos += 1
                if (pos > 10):
                    break


if __name__ == '__main__':
    SimpleCF(k=10)
