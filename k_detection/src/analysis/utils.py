def clusterFinder(findMe, labels, data):
    cluster = []
    for j in range(labels.shape[0]):
        if labels[j] == findMe:
            cluster.append(data[j])
    return cluster


def outerPointsFromTheCluster(findMe, labels, data):
    outerPoints = []
    for j in range(labels.shape[0]):
        if labels[j] != findMe:
            outerPoints.append(data[j])
    return outerPoints