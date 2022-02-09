import numpy as np
from scipy.spatial.distance import cdist

class Silhouette:
    def __init__(self, metric: str = "euclidean"):
        """
        inputs:
            metric: str
                the name of the distance metric to use
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
        k_labs = np.unique(y)
        s_val_all = np.zeros(len(X))
        for k_val in k_labs:
            len_k = len(X[y==k_val])
            for idx, pt in zip(np.where(y==k_val)[0], X[y==k_val]):
                dist_mat_same = cdist([pt], X[y==k_val], self._metric)[0]
                a_val = 1/(len_k-1)*sum(dist_mat_same)
                diff_k_val = np.delete(k_labs, k_val)
                tmp_b=[]
                for k_val_diff in diff_k_val:
                    len_k_diff = len(X[y==k_val_diff])
                    diff_sum = sum(cdist([pt], X[y==k_val_diff], self._metric)[0])
                    tmp_b.append((1/len_k_diff)*diff_sum)
                b_val = min(tmp_b)
                s_val = (b_val-a_val)/(max(a_val, b_val))
                s_val_all[idx] = s_val
        return s_val_all
   


