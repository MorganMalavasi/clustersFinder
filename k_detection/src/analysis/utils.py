import os
import csv

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
    
    path_to_file_samples = 'analysis/cqcluster/k_means_input.csv'
    
    # printMatrix(samples, labels)
    
    with open(path_to_file_samples, "w") as csvfile:
        csvwriter = csv.writer(csvfile)
        
        for i in range (samples.shape[0]):
            if i == 0:
                size = samples.shape[1]
                write_col_of_Data_frame(csvwriter, size)
            
            row = samples[i,:]
            csvwriter.writerow(row)

    path_to_file_labels = 'analysis/cqcluster/labels_input.csv'

    with open(path_to_file_labels, "w") as csvfile:
        csvwriter = csv.writer(csvfile)

        for i in range(labels.shape[0]):
            if i == 0:
                size = 1
                write_col_of_Data_frame(csvwriter, size)
            
            row = [labels[i] + 1]
            csvwriter.writerow(row)
    

    return 

def deleteFile():
    os.remove("analysis/cqcluster/k_means_input.csv")
    os.remove("analysis/cqcluster/labels_input.csv")
    return 

def printMatrix(samples, labels):
    for i in range(samples.shape[0]):
        print("{0} {1}".format(samples[i,:], labels[i]))
    
    return 

def write_col_of_Data_frame(csvwriter, size):
    myTuple = ()
    for k in range(size):
        # TODO -> bug, it trasforms the two digits numbers in just one digit number 
        myTuple = myTuple + (str(k), ) 
        # print(str(k))
    print(myTuple)
    x = "".join(myTuple)
    print(x)
                
    csvwriter.writerow(x)