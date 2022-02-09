# Write your k-means unit tests here
from cluster import KMeans
from cluster import make_clusters
import unittest


def test_descend_errors():
    '''
    this unit test assures that each itteration of the kmeans fit produces a lower
    error than the prior itteration
    '''
    mat,_ = make_clusters(k=3)
    kmeans = KMeans(k=3)
    kmeans.fit(mat)
    all_errs = list(kmeans.get_all_errors())
    all_errs_cp = all_errs.copy()
    all_errs_cp.sort(reverse=True)
    assert all_errs == all_errs_cp
    
def test_output():
    '''
    test that the output labels is the same length as the input matrix, and the labels
    all all acceptable values of k
    '''
    k=10
    mat,_ = make_clusters(k)
    kmeans = KMeans(k)
    kmeans.fit(mat)
    labels = kmeans.predict(mat)
    assert len(labels) == len(mat)
    for l in labels:
        assert l in range(k)
        


def test_extreme():
    '''
    testing higher values of k, n
    '''
    mat,_ = make_clusters(k=3000, n = 20000)
    kmeans = KMeans(k=300)
    kmeans.fit(mat)
    kmeans.predict(mat)
    pass


class raiseTest_kval(unittest.TestCase):
    '''
    testing edge cases that SHOULD throw errors (such as trying to make k = 0, 
    or trying to use more k's than there are number of points.
    '''
    def testraise1(self):
         self.assertRaises(ValueError, KMeans, 0)
    
    def testraise2(self):
        mat,_ = make_clusters()
        kmeans = KMeans(k=1000)
        with self.assertRaises(ValueError):
            kmeans.fit(mat)
            
    if __name__ == "__main__":
        unittest.main()
    
    
    



    
    