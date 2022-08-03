import numpy as np
import sys
from analysis.utils import clusterFinder, outerPointsFromTheCluster
from scipy.spatial.distance import euclidean

def separationindex(data, labels):
    p = 0.1
    listOfpnk = []
    
    # numerator
    sumNumerator = 0.0
    for i in range(int(max(labels) + 1)):
        objectsInTheCluster = clusterFinder(i, labels, data)
        objectsInTheCluster = np.asarray(objectsInTheCluster)

        objectsOutsideTheCluster = outerPointsFromTheCluster(i, labels, data)
        objectsOutsideTheCluster = np.asarray(objectsOutsideTheCluster)

        listOfdistances = []
        for k in range(objectsInTheCluster.shape[0]):
            min = sys.maxsize
            for j in range(objectsOutsideTheCluster.shape[0]):
                distanceBetweenTheTwoPoints = euclidean(objectsInTheCluster[k,:], objectsOutsideTheCluster[j,:])
                if distanceBetweenTheTwoPoints < min:
                    min = distanceBetweenTheTwoPoints
            listOfdistances.append(min)
        
        arrayOfDistances = np.asarray(listOfdistances)
        arrayOfDistancesSorted = np.sort(arrayOfDistances)

        pnk = int(p * arrayOfDistancesSorted.shape[0])
        listOfpnk.append(pnk)

        sum = 0.0
        for h in range(pnk):
            sum += arrayOfDistancesSorted[h]
        sumNumerator += sum

    # denominator
    sumDenominator = 0.0
    for l in range(len(listOfpnk)):
        sumDenominator += listOfpnk[l]

    return sumNumerator / sumDenominator
