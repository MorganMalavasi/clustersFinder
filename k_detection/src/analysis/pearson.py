import re
import numpy as np
from scipy import stats

def pearson_index(data, labels, matrixOfDissimilarities):

    # vectorization of the matrix
    size = ((matrixOfDissimilarities.shape[0] * matrixOfDissimilarities.shape[1]) // 2) - (matrixOfDissimilarities.shape[0] // 2)
    vectorDissimilarities = np.zeros(size)
    vectorOf1 = np.zeros(size)
    counter = 0
    for i in range (matrixOfDissimilarities.shape[0]):
        for j in range(i + 1, matrixOfDissimilarities.shape[1]):
            vectorDissimilarities[counter] = matrixOfDissimilarities[i, j]
            if labels[i] != labels[j]:
                vectorOf1[counter] = counter
            else:
                vectorOf1[counter] = 0
            counter += 1

    res = stats.pearsonr(vectorDissimilarities, vectorOf1)
    return res[1]