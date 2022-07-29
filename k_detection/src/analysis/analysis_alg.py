import sys
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans


class internal_analysis:
    
    minSizeCluster = 2 # min number of clusters
    maxSizeCluster = 15 # max number of clusters 
    size = maxSizeCluster - minSizeCluster 
    kMeans_ = None
    kMeans_list = None   

    

    def k_means_nrClusters(self, numberK, samples):
        return KMeans(n_clusters=numberK).fit(samples)

    def k_means_array(self, data):
        self.kMeans_list = []
        for i in range(self.minSizeCluster, self.maxSizeCluster):
            self.kMeans_list.append(self.k_means_nrClusters(numberK=i, samples=data))
        
    # ************************* silhouette ************************************
    def k_means_silhouette_nrClusters_defined(self, nr_clusters, data):
        if self.kMeans_ == None:
            self.kMeans_ = self.k_means_nrClusters(numberK = nr_clusters, samples = data)
        # compute silhoutte score 
        score = metrics.silhouette_score(data, self.kMeans_.labels_, metric="euclidean")
        print(score)

    def k_means_silhouette(self, data):
        maxSilhoutte = -(sys.maxsize)
        
        if self.kMeans_list == None:
            self.k_means_array(data)

        for i in range(len(self.kMeans_list)):
            score = metrics.silhouette_score(data, self.kMeans_list[i].labels_, metric="euclidean")
        
            if score > maxSilhoutte:
                maxSilhoutte = score
        
        print(maxSilhoutte)
                
    def circleClustering_silhouette(self, data, labels):
        score = metrics.silhouette_score(data, labels, metric="euclidean")
        print(score)

    # ************************* ch_calinski_harabasz **************************
    def k_means_calinski_harabasz_nrClusters_defined(self, nr_clusters, data):
        if self.kMeans_ == None:
            self.kMeans_ = self.k_means_nrClusters(numberK = nr_clusters, samples = data)
        # compute silhoutte score 
        score = metrics.calinski_harabasz_score(data, self.kMeans_.labels_)
        print(score)

    def k_means_calinski_harabasz(self, data):
        maxSilhoutte = -(sys.maxsize)
        
        if self.kMeans_list == None:
            self.k_means_array(data)

        for i in range(len(self.kMeans_list)):
            score = metrics.calinski_harabasz_score(data, self.kMeans_list[i].labels_)
        
            if score > maxSilhoutte:
                maxSilhoutte = score
        
        print(maxSilhoutte)
                
    def circleClustering_calinski_harabasz(self, data, labels):
        score = metrics.calinski_harabasz_score(data, labels)
        print(score)