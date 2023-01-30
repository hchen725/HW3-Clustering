import numpy as np
import random
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        self.check_init_vals(k, tol, max_iter)

        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def check_init_vals(self, k, tol, max_iter):
        # Check k values
        if k <= 0:
            raise ValueError("k must be a positive integer")
        if tol <= 0:
            raise ValueError("tol must be positive")
        if max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")
        
    def check_mat(self, mat):
        num_obs = mat.shape[0] # Total number of observations
        num_features = mat.shape[1] # Total number of features
        
        # Check the observations
        if num_obs == 0:
            raise ValueError("There are no observations present in the dataset")
        if num_obs < self.k:
            raise ValueError("The number of observations must be greater than k")

        # Check the features
        if num_features == 0:
            raise ValueError("There are no features present in the dataset")
        

    # def get_cluster_label(self, mat: np.ndarray) -> np.ndarray:
    #     dists = cdist(mat, self.centroids) # Calculate the distance of each point to all other centroids
    #     cluster_labels = np.argmin(dists, axis = 1)  # Get the index of the smallest distance
    #     return(cluster_labels)


    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        # Check the matrix input
        self.check_mat(mat)
        self.mat = mat
        num_obs = mat.shape[0]

        print("Fitting " + str(num_obs) + " observations to " + str(self.k) + " clusters")
        
        # Randomly select k number of points from the mat as the starting centroids
        self.centroids = mat[random.sample(range(0, num_obs), self.k)]

        # Initialize iteration counter and error record
        num_iter = 0
        fit_error = np.inf

        while num_iter < self.max_iter and fit_error > self.tol:

            # Calculate the distance from each observation to each centroid
            # Assign each observation to nearest centroid and update their membership
            self.cluster_labels = self.predict(mat)
            # Find the centroid of the new cluster
            self.old_centroids = self.centroids 
            self.centroids = self.get_centroids()

            # Calculate and update the error
            fit_error = self.get_error(self.old_centroids, self.centroids)

            # Update iteration counter
            num_iter += 1
        

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        dists = cdist(mat, self.centroids) # Calculate the distance of each point to all other centroids
        cluster_labels = np.argmin(dists, axis = 1)  # Get the index of the smallest distance
        return(cluster_labels)


    def get_error(self, old, new) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        mse = ((new - old)**2).mean()
        return (mse)

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        # Initialize cluster centroids of k x m to hold values for each cluster k and each feature m
        cluster_centroids = np.zeros(shape = (self.k, self.mat.shape[1]))
        for cluster in range(0, self.k):
            cluster_centroids[cluster, :] = np.mean(self.mat[self.cluster_labels == cluster, :], axis = 0)
        return(cluster_centroids)
