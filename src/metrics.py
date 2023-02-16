import random
from itertools import product
from math import sqrt
from typing import List

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

def prediction_coverage(predicted, catalog):
    """
    Computes the prediction coverage for a list of recommendations
    Parameters
    ----------
    predicted : a list of lists
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    catalog: list
        A list of all unique items in the training data
        example: ['A', 'B', 'C', 'X', 'Y', Z]
    Returns
    ----------
    prediction_coverage:
        The prediction coverage of the recommendations
    ----------
    Metric Defintion:
    Ge, M., Delgado-Battenfeld, C., & Jannach, D. (2010, September).
    Beyond accuracy: evaluating recommender systems by coverage and serendipity.
    In Proceedings of the fourth ACM conference on Recommender systems (pp. 257-260). ACM.
    """
    predicted_flattened = [p for sublist in predicted for p in sublist if p in catalog]
    prediction_coverage = len(set(predicted_flattened)) / len(catalog)
    return prediction_coverage

def recall(predictions, labels):
    label_set = set(labels)
    count = 0
    for p in predictions:
        if p in label_set:
            count += 1
    return count / len(label_set)


def reciprocal_rank(predictions, labels):
    label_set = set(labels)
    for r, p in enumerate(predictions):
        if p in label_set:
            return 1 / (r + 1)
    return 0.0

def average_precision(predictions, labels):
    label_set = set(labels)
    correct_predictions = 0
    running_sum = 0
    for i, pred in enumerate(predictions):
        k = i+1 # our rank starts at 1
        if pred in label_set:
            correct_predictions += 1
            running_sum += correct_predictions/k
    return running_sum/len(label_set)


