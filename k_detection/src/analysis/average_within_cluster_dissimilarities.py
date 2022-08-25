import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from analysis.utils import clusterFinder

def sumDiss(matrix):
    sum = 0
    for i in range(matrix.shape[0]):
        for j in range(i+1, matrix.shape[1]):
            sum += matrix[i, j]

    return sum

def average_within_cluster_dissimilarities(data, labels):
    maxLabel = int(max(labels))
    total = 0
    for i in range(maxLabel):
        cls = clusterFinder(i, labels, data)
        cluster_array = np.asarray(cls)

        matrixOfDistances = euclidean_distances(cluster_array, cluster_array)
        sumDissimilarities = sumDiss(matrixOfDistances)
        if (cluster_array.shape[0] > 1):
            sumDissimilarities = sumDissimilarities / (cluster_array.shape[0] - 1)

        total += sumDissimilarities
    
    total = total / data.shape[0]

    return total
