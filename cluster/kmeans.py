import numpy as np
from scipy.spatial.distance import cdist

class KMeans:
    def __init__(
            self,
            k: int,
            metric: str = "euclidean",
            tol: float = 1e-6,
            max_iter: int = 100):
        """
        inputs:
            k: int
                the number of centroids to use in cluster fitting
            metric: str
                the name of the distance metric to use
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        self.k = k
        self.metric = metric
        self.tol = tol
        self.max_iter = max_iter
        self._err = np.inf
        
    
    def fit(self, mat: np.ndarray):
        """
        fits the kmeans algorithm onto a provided 2D matrix

        inputs: 
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """

        ctrs_idx = np.random.choice(len(mat), self.k, replace=False) # find k random indices to assigns as centroids
        centroids = mat[ctrs_idx, :] # make these points the centroids
        dist_to_centroids = cdist(mat, centroids, self.metric) # find dist of each point to each of the centroids
        closest_centroid = np.array([np.argsort(d)[0] for d in dist_to_centroids]) # find the closest centroid to each point

        itr=0
        while itr < max_iter: # max iterations to go for before exiting
            centroids = []
            for k_idx in range(self.k): # for each k center
                centroids.append(mat[closest_centroid==k_idx].mean(axis=0)) # find a new centroid at the middle of the assigned points
            dist_to_centroids = cdist(mat, centroids, self.metric) # find dist of each point to each of the new centroids
            closest_centroid = np.array([np.argsort(d)[0] for d in dist_to_centroids]) # find the closest centroid to each point
            itr+=1
            new_err = calc_mse(mat, self.k, closest_centroid) # calculate the error of this round
            if self._err - new_err < tol: # if new error is the value of tol less than the previous error
                self._err = new_err
                break
            self._err = new_err

        self._centroids = centroids  

    def _calc_mse(self, mat, closest_centroid):
        """
        calculates the mean square error of a matrix and the centroids
        
        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features             
            closest_centroid:
                the closest centroid to each point in mat
        returns:
            float
                the mean square error score
        """
        sq_err = []
        for k_idx in range(self.k):
            n = len(mat[closest_centroid==k_idx]) # denominator of mse (# of points assigned to that k)
            num = sum(np.square(cdist(mat[closest_centroid==k_idx], # numerator of mse (square of distances of points to centroid)
                                    [centroids[k_idx]])))
            sq_err.append((num/n)[0])

        return sum(sq_err) # final mse is sum of this for all k's
    
    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        predicts the cluster labels for a provided 2D matrix

        inputs: 
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        dist_to_centroids = cdist(mat, self._centroids) # use the centroids determined in fit
        closest_centroid = np.array([np.argsort(d)[0] for d in dist_to_centroids]) # find closest centroid to each point
        return closest_centroid
    
    def get_error(self) -> float:
        """
        returns the final squared-mean error of the fit model

        outputs:
            float
                the squared-mean error of the fit model
        """
        return self._err

    def get_centroids(self) -> np.ndarray:
        """
        returns the centroid locations of the fit model

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        return self._centroids

