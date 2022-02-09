import numpy as np
from scipy.spatial.distance import cdist

class KMeans:
    def __init__(
            self,
            k: int,
            metric: str = "euclidean",
            tol: float = 1e-6,
            max_iter: int = 100,
            ):
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
        if self.k==0:
            raise ValueError('k value cannot be zero')
        self.metric = metric
        self.tol = tol
        self.max_iter = max_iter
        self._err = np.inf
        self._all_scores = []
    
    
    
    
    def fit(self, mat: np.ndarray):
        """
        fits the kmeans algorithm onto a provided 2D matrix

        inputs: 
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        if self.k>len(mat):
            raise ValueError('more cluster centers than points')
            
        closest_centroid = np.array([np.random.randint(0,self.k) for e in range(len(mat))]) # randomly assign each point in mat to a k
        itr=0
        while itr <= self.max_iter: # max iterations to go for before exiting
            centroids = []
            for k_idx in range(self.k): # for each k center
                # find a new centroid at the middle of the assigned points
                centroids.append(mat[closest_centroid==k_idx].mean(axis=0)) 
            self._centroids = centroids
            dist_to_centroids = cdist(mat, centroids, self.metric) # find dist of each point to each of the new centroids
            closest_centroid = np.array([np.argmin(d) for d in dist_to_centroids]) # find the closest centroid to each point
            itr+=1

            new_err = self._calc_mse(mat, np.array(closest_centroid)) # calculate the error of this round
            if len(self._all_scores) > 0:
                self._all_scores.append(new_err)
            else:
                self._all_scores= [new_err]
            if abs(self._err - new_err) < self.tol: # if new error is the value of tol less than the previous error
                self._err = new_err
                break
            else:
                self._err = new_err
            
        

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
            # numerator of mse (square of distances of points to centroid)
            num = sum(np.square(cdist(mat[closest_centroid==k_idx],[self._centroids[k_idx]], self.metric)))
            if n>0:
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
    
    def get_all_errors(self) -> np.ndarray:
        """
        returns all the MSE error scores for each itteration

        outputs:
            np.ndarray
                an array of each score
        """
        return self._all_scores

    def get_centroids(self) -> np.ndarray:
        """
        returns the centroid locations of the fit model

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        return self._centroids

