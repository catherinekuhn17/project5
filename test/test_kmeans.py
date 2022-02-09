# Write your k-means unit tests here
from kmeans import KMeans
from utils import make_clusters

def test_descend_errors():
    '''
    this unit test assures that each itteration of the kmeans fit produces a lower
    error than the prior itteration
    '''
    mat,_ = make_clusters()
    kmeans = KMeans(k=3)
    kmeans.fit(mat)
    all_errs = list(kmeans.get_all_errors())
    all_errs_cp = all_errs.copy()
    all_errs_cp.sort(reverse=True)
    assert all_errs == all_errs_cp
    
def all_k():
    '''
    testing that output labels contain all values of k
    '''
    pass
def test_edge_cases():
    '''
    '''
    # testing for when k=0 (which should not happen)
    pass

def test_extreme():
    '''
    testing extrame values of k
    '''
    pass


    
    