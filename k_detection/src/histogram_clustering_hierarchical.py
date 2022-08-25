from random import sample
import numpy as np
from anytree import AnyNode
from anytree.exporter import DotExporter
from sklearn.neighbors._nearest_centroid import NearestCentroid
import sys


def getClustersFromHistogram(heights, nbins):
    # start alg
    # print(" - create hierarchical tree")
    tree = createHierarchicalTree(heights, nbins)
    # print(" - creation of the file tree.png with the tree")
    
    # print the tree in tree.png
    # DotExporter(tree).to_picture("tree.png")

    detectClusters(tree)        
    newTree = createTreeOfClusters(tree, nbins.shape[0]-1)
    # DotExporter(newTree).to_picture("clusters.png")
    
    # each cluster is a tuple that indicates the number of the cluster and the interval of membership
    clusters = searchClusters(newTree)
    return clusters


# ******************************* creation of the tree of reasearch *******************************************

# return the correct parent between all the parents 
def correctParent(parents, intervalOfNode, maxLength):
    start = 0
    end = maxLength
    parent = None

    intervalOfNodeStart = intervalOfNode[0]
    intervalOfNodeEnd = intervalOfNode[1]

    for i in range(len(parents)):
        intervalParent = parents[i].interval
        intervalParentStart = intervalParent[0]
        intervalParentEnd = intervalParent[1]
        
        if intervalOfNodeStart >= intervalParentStart and intervalOfNodeEnd <= intervalParentEnd:
            if intervalOfNodeStart >= start and intervalOfNodeEnd <= end:
                start = intervalOfNodeStart
                end = intervalOfNodeEnd
                parent = parents[i]
    return parent


def addToParent(parents, interval, area, maxLength):
    intervalChild = interval
    areaChild = area
    _parent_ = correctParent(parents, intervalChild, maxLength)
    id = _parent_.name
    n_children = len(_parent_.children)
    newId = id + "_" + str(n_children)
    newNode = AnyNode(name=newId, interval = intervalChild, area = areaChild, clusters = 0, parent=_parent_)
    parents.append(newNode)


def createHierarchicalTree(heights, nbins):

    tree = AnyNode(name="0", interval = (0.0, nbins.shape[0]-1), area = 0.0, clusters = 0)
    
    parents = []
    parents.append(tree)

    listOfheightsToCheck = sorted(set(heights))
    # len(listOfheightsToCheck) 
    lastHeight = 0.0
    for i in range(len(listOfheightsToCheck) ):
        if listOfheightsToCheck[i] != 0.0:
            checkHeight = listOfheightsToCheck[i]

            j = 0
            while j < (nbins.shape[0]-1): 
                if heights[j] >= checkHeight:
                    start = j
                    end = nbins.shape[0]-1
                    while j < (nbins.shape[0]-1) and heights[j] >= checkHeight:
                        j = j+1
                        end = j
                    # add a new node
                    interval = (start, end)
                    
                    area = (end - start) * (listOfheightsToCheck[i] - lastHeight)
                    addToParent(parents, interval, area, nbins.shape[0])
                else: 
                    j = j+1
            lastHeight = listOfheightsToCheck[i]

    return tree



# ******************************* detect the cut where it starts two or more clusters *************************

# visit in depth
def returnTheAreaUnderThisNode(root):
    stack = []
    stack.append(root)
    totArea = 0.0

    while len(stack) > 0:
        currentNode = stack.pop()
        totArea += currentNode.area

        # put children on the stack
        children = currentNode.children
        for i in range(len(children)):
            stack.append(children[len(children)-1 - i])

    return totArea

def detectClusters(tree):
    
    # traverse all the tree and compute the areas
    node = tree
    stack = []
    stack.append(node)
    
    while len(stack) > 0:
        currentNode = stack.pop()

        if len(currentNode.children) > 1:
            
            # area of the upper part before a cut
            areaOfTheUpperPart = currentNode.area
            parentNode = currentNode.parent
            while parentNode != None and len(parentNode.children) <= 1:
                areaOfTheUpperPart += parentNode.area
                parentNode = parentNode.parent

            # area of the part under the cut
            areaOfTheChildren = 0.0
            for i in range(len(currentNode.children)):
                areaOfTheChildren += returnTheAreaUnderThisNode(currentNode.children[i])
            
            # check -> if the area of children is bigger than the area of upper part before a cut
            # then there are a number of clusters equal the number of the children
            if areaOfTheChildren > areaOfTheUpperPart:
                currentNode.clusters = len(currentNode.children)
  
        
        # put children on the stack
        children = currentNode.children
        for i in range(len(children)):
            stack.append(children[len(children)-1 - i])


# ******************************* create the tree of clusters and detect their number *************************

def createTreeOfClusters(tree, size_bins):

    interval = (0.0, size_bins)
    newTree = AnyNode(name="[" + str(interval[0]) + " - " + str(interval[1]) + "]", interval = interval)
    
    stackOfParents = []
    stackOfParents.append(newTree)

    node = tree
    stack = []
    stack.append(node)

    while len(stack) > 0:
        currentNode = stack.pop()
        if currentNode.clusters > 0:
            children = currentNode.children
            for i in range(len(children)):
                currentChildren = children[i]
                _parent_ = correctParent(stackOfParents, currentChildren.interval, size_bins)
                newNode = AnyNode(name="[" + str(currentChildren.interval[0]) + " - " + str(currentChildren.interval[1]) + "]", interval = currentChildren.interval, parent = _parent_)
                stackOfParents.append(newNode)
            
        # put children on the stack
        children = currentNode.children
        for i in range(len(children)):
            stack.append(children[len(children) - 1 - i])        

    return newTree


def searchClusters(tree):
    counterClusters = 0
    clusters = []

    node = tree
    stack = []
    stack.append(node)

    while len(stack) > 0:
        currentNode = stack.pop()

        # if it hasn't children it means it is a leaf (a cluster)
        if len(currentNode.children) == 0:
            clusters.append((counterClusters, currentNode.interval))
            counterClusters += 1

        # put children on the stack
        children = currentNode.children
        for i in range(len(children)):
            stack.append(children[len(children) - 1 - i])

    return clusters
    


def labelTheSamples(samples, theta, clusters, bins):
    # TODO -> rivedere, c'è qualche errore nella vicinanza perchè a volte punti vicini sono labellizati in modo sbagliato 
    label = np.empty(samples.shape[0])
    
    for i in range(theta.shape[0]):
        value = theta[i]
        distance = sys.maxsize
        labelFound = None

        for j in range(len(clusters)):
            cluster = clusters[j]
            
            nr_cluster = cluster[0]
            intervalIndex = cluster[1]

            distanceX = abs(value - bins[intervalIndex[0]])
            distanceY = abs(value - bins[intervalIndex[1]])
            minDistance = min(distanceX, distanceY)

            if minDistance < distance:
                distance = minDistance
                labelFound = nr_cluster
        
        label[i] = labelFound
        
    return label

def centroidsFinder(samples, labels):
    clf = NearestCentroid().fit(samples, labels)
    return clf.centroids_


'''
def searchClusters(tree):

    tot_nr_clusters = 0
    nr_cut = 0

    node = tree
    stack = []
    stack.append(node)

    while len(stack) > 0:
        currentNode = stack.pop()
        if currentNode.clusters > 0:
            tot_nr_clusters += currentNode.clusters
            nr_cut += 1
            print("interval = {0}, clusters = {1}".format(currentNode.interval, currentNode.clusters))

        
        # put children on the stack
        children = currentNode.children
        for i in range(len(children)):
            stack.append(children[len(children) - 1 - i])


    return tot_nr_clusters - nr_cut + 1
'''