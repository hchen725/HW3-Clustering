import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self, metric: str = "euclidean" ):
        """
        inputs:
            none
        """
        self._metric = metric

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """

        # X = matrix
        # Y = labels
        # Return array of scores

        num_obs = X.shape[0] # Number of observations in X 
        cluster_list = np.unique(y) # get the unique clusters
        num_clusters = len(cluster_list) # number of clusters

        # Calculate distance between all pairs of obs
        pairwise_dists = cdist(X, X, metric = self._metric)
        sil_scores = np.zeros(num_obs)

        # loop through observations
        for obs in range(0, num_obs):
            current_cluster = y[obs] # Get current cluster
            cluster_size = len(y[y == current_cluster]) # Get num of obs in cluster

            if cluster_size == 1:
                sil_scores[obs] = 0
            else:
                # Calculate a
                a = self._calculate_a(pairwise_dists, y, current_cluster, obs)
            
                # Calculate b
                b = self._calculate_b(pairwise_dists, y, current_cluster, obs, cluster_list)
                sil_scores[obs] = (b-a)/max(a,b)        

        return sil_scores

    def _calculate_a(self, pariwise_dists, y, current_cluster, obs):
        cluster_size = len(y[y == current_cluster])
        same_cluster_dists = pariwise_dists[y == current_cluster][:,obs]
        a = np.sum(same_cluster_dists)/(cluster_size - 1)
        return a

    def _calculate_b(self, pairwise_dists, y, current_cluster, obs, cluster_list):
        other_clusters = np.delete(cluster_list, current_cluster)
        b = np.inf
        for c in other_clusters:
            other_cluster_dists = pairwise_dists[y == c][:, obs]
            _b = np.mean(other_cluster_dists)
            b = min(b, _b)
        return b

    