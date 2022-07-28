import sys
from sklearn import metrics
from sklearn.cluster import KMeans

def k_means_silhouette_nrClusters_defined(nr_clusters, data):
    kMeansWithHelp = KMeans(n_clusters=nr_clusters).fit(data)
    # compute silhoutte score 
    score = metrics.silhouette_score(data, kMeansWithHelp.labels_, metric="euclidean")
    print(score)

def k_means_silhouette(data):
    maxSilhoutte = -(sys.maxsize)
    for i in range(2, 15):
        kMeans_ = KMeans(n_clusters=i).fit(data)
        score = metrics.silhouette_score(data, kMeans_.labels_, metric="euclidean")
        if score > maxSilhoutte:
            maxSilhoutte = score
    print(maxSilhoutte)
            
def circleClustering_silhouette(data, labels):
    score = metrics.silhouette_score(data, labels, metric="euclidean")
    print(score)