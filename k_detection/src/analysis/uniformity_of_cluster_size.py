import math
import numpy as np
from cmath import log
from analysis.utils import clusterFinder

def entropy(data, labels):
    nr_clusters = int(max(labels) + 1)

    sum = 0.0
    for i in range(nr_clusters):
        cls = clusterFinder(i, labels, data)
        cls = np.asarray(cls)
        sum += (cls.shape[0] / data.shape[0]) * math.log2(cls.shape[0] / data.shape[0])

    return -(sum)