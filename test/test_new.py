# write your silhouette score unit tests here
import pytest
from cluster import KMeans
from cluster import make_clusters
from cluster import Silhouette
import unittest

def score_range():
    '''
    unit test to assert that all scores fall between 0 and 1
    '''
    mat,_ = make_clusters(k=3)
    kmeans = KMeans(k=3)
    kmeans.fit(mat)
    labels = kmeans.predict(mat)
    silhouette = Silhouette()
    sil_score = silhouette.score(mat, labels)
    
    assert all(i <= 1 for i in sil_score)
    assert all(i >= 0 for i in sil_score)
    
    
def score_range():
    '''
    unit test to assert that all scores fall between 0 and 1
    '''
    mat,_ = make_clusters(k=3)
    kmeans = KMeans(k=3)
    kmeans.fit(mat)
    labels = kmeans.predict(mat)
    silhouette = Silhouette()
    sil_score = silhouette.score(mat, labels)
    
    assert len(sil_score) == len(mat)

