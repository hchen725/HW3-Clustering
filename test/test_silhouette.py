# write your silhouette score unit tests here
import pytest
from cluster import kmeans, silhouette, utils
import numpy as np
import random
import math
from sklearn.metrics import silhouette_score

# Test to make sure the silhouette score is between -1 and 1
def kmean_example():
    N = 500
    mat, _ = utils.make_clusters(n = N, k = 3)

    # Split the data into training and testing sets
    idx = list(range(0,N))
    train_idx = random.sample(idx, math.floor(N/4)) # Training set
    test_idx = list(set(idx) - set(train_idx)) # Testing set

    # Get training mat and test mat
    mat_train = mat[train_idx, :]
    mat_test = mat[test_idx, :]

    # Initialize kmeans and train on training data
    km = kmeans.KMeans(k = 3)

    return(km, mat_test, mat_train)

def test_silhouette_range():
    km, mat_test, mat_train = kmean_example()
    km.fit(mat_train)
    # Get predicted labels of the test set
    labels_pred = km.predict(mat_test)

    # Calculate silhouette scores
    sil = silhouette.Silhouette()
    sil_scores = sil.score(mat_test, labels_pred)

    assert (sil_scores.all() >= -1) and (sil_scores.all() <= 1)

def test_compare_to_sklean():
    km, mat_test, mat_train = kmean_example()
    km.fit(mat_train)
    # Get predicted labels of the test set
    labels_pred = km.predict(mat_test)

    # Calculate silhouette scores
    sil = silhouette.Silhouette()

    sil_scores = sil.score(mat_test, labels_pred)
    sklearn_scores = silhouette_score(mat_test, labels_pred)
    assert np.isclose(np.mean(sil_scores), sklearn_scores)