# Write your k-means unit tests here
import pytest
from cluster import kmeans, silhouette, utils
import numpy as np
from sklearn.metrics import mean_squared_error


# Test initialization with negative k and 0 k
def test_check_k_neg():
    with pytest.raises(ValueError, match = "k must be a positive integer"):
        kmeans.KMeans(k = -10)

def test_check_k_zero():
    with pytest.raises(ValueError, match = "k must be a positive integer"):
        kmeans.KMeans(k = 0)  

# Test initiazliation with negative tol
def test_check_tol_neg():
    with pytest.raises(ValueError, match = "tol must be positive"):
        kmeans.KMeans(k = 10, tol = -5)  

# Test initialization with negative iterations
def test_check_neg_iter():
    with pytest.raises(ValueError, match = "max_iter must be a positive integer"):
        kmeans.KMeans(k = 10, max_iter = -5)  

def test_mat_empty_obs():
    empty_mat = np.empty((0, 10))
    km = kmeans.KMeans(k = 5)
    with pytest.raises(ValueError, match = "There are no observations present in the dataset"):
        km.check_mat(empty_mat)

def test_mat_empty_feat():
    empty_mat = np.empty((10, 0))
    km = kmeans.KMeans(k = 5)
    with pytest.raises(ValueError, match = "There are no features present in the dataset"):
        km.check_mat(empty_mat)


def test_mat_small():
    small_mat, _ = utils.make_clusters(n = 4)
    km = kmeans.KMeans(k = 5)
    with pytest.raises(ValueError, match = "The number of observations must be greater than k"):
        km.check_mat(small_mat)

def test_compare_mse_sklearn():
    km = kmeans.KMeans(k = 5)
    old = np.array([[2,2,3], [3,4,5]])
    new = np.array([[1,3,4], [3,4,5]])
    mse = km.get_error(old, new)
    mse_sk = mean_squared_error(old, new)
    assert mse == mse_sk