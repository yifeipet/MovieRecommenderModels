# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))

from MovieLens import MovieLens
from surprise import SVD, SVDpp, NormalPredictor
from evaluate import Evaluator

import random
import numpy as np


class SVDBakeoff:
    def __init__(self):
        #set seeds for repeatable results
        np.random.seed(0)
        random.seed(0)
        #load function extracted into 3 steps
        ml = MovieLens()
        evaluationData = ml.loadMovieLensLatestSmall()
        rankings = ml.getPopularityRanks()
        # Construct an Evaluator to, you know, evaluate them
        evaluator = Evaluator(evaluationData, rankings)
        # SVD
        svd = SVD()
        evaluator.AddAlgorithm(svd, "SVD")
        # SVD++
        SVDPlusPlus = SVDpp()
        evaluator.AddAlgorithm(SVDPlusPlus, "SVD++")
        # Just make random recommendations
        Random = NormalPredictor()
        evaluator.AddAlgorithm(Random, "Random")
        # Fight!
        evaluator.Evaluate(False)
        evaluator.SampleTopNRecs(ml)


if __name__ == '__main__':
    SVDBakeoff()
