import csv
import numpy as np

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

def createFile(samples, labels):
    
    path_to_file = 'analysis/cqcluster/k_means_input.csv'
    # open(path_to_file, 'w')

    printMatrix(samples, labels)
    
    with open(path_to_file, "w") as csvfile:
        csvwriter = csv.writer(csvfile)
        
        for i in range (samples.shape[0]):
            if i == 0:
                size = samples.shape[1]
                myTuple = ()
                for k in range(size):
                    myTuple = myTuple + (str(k), )
                x = "".join(myTuple)
                          
                csvwriter.writerow(x)
            
            row = samples[i,:]
            csvwriter.writerow(row)

    '''
    with open ...

                if i == 0:
                size = samples.shape[1]
                myTuple = ()
                for k in range(size + 1):
                    myTuple = myTuple + (str(k), )
                x = "".join(myTuple)
                          
                csvwriter.writerow(x)
    '''

    return 

def deleteFile():

    return 

def printMatrix(samples, labels):
    for i in range(samples.shape[0]):
        print("{0} {1}".format(samples[i,:], labels[i]))
    
    return 