import numpy as np
from anytree import AnyNode, RenderTree
from anytree.exporter import DotExporter


def getClustersFromHistogram(heights, nbins):
    numberOfClusters = 0

    # start alg
    tree = AnyNode(name="0", interval = (0.0, nbins.shape[0]), area = 0.0, areaAccumulated = 0.0, superiorArea = 0.0, clusters = 0)
    print(" - create hierarchical tree")
    createHierarchicalTree(tree, heights, nbins)
    print(" - creation of the file tree.png with the tree")
    
    # print the tree in tree.png
    DotExporter(tree).to_picture("tree.png")
    
    sumCumulativeArea(tree)
    leaves = []
    detectLeaves(tree, leaves)
    leaves.reverse()
    # search the clusters


    return numberOfClusters



# return the correct parent between all the parents 
def correctParent(parents, intervalOfNode, maxLength):
    start = 0
    end = maxLength
    parent = None
    for i in range(len(parents)):
        intervalParent = parents[i].interval
        intervalParentStart = intervalParent[0]
        intervalParentEnd = intervalParent[1]

        intervalOfNodeStart = intervalOfNode[0]
        intervalOfNodeEnd = intervalOfNode[1]
        
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
    newNode = AnyNode(name=newId, interval = intervalChild, area = areaChild, areaAccumulated = 0.0, superiorArea = 0.0, clusters = 0, parent=_parent_)
    parents.append(newNode)


def createHierarchicalTree(tree, heights, nbins):
    
    parents = []
    parents.append(tree)

    listOfheightsToCheck = sorted(set(heights))
    # len(listOfheightsToCheck) 
    for i in range(len(listOfheightsToCheck) ):
        if listOfheightsToCheck[i] != 0.0:
            checkHeight = listOfheightsToCheck[i]

            j = 0
            while j < (nbins.shape[0]-1): 
                if heights[j] >= checkHeight:
                    start = j
                    while j < (nbins.shape[0]-1) and heights[j] >= checkHeight:
                        j = j+1
                        end = j
                    # add a new node
                    interval = (start, end)
                    area = (end - start) * listOfheightsToCheck[i]
                    addToParent(parents, interval, area, nbins.shape[0])
                else: 
                    j = j+1


def sumCumulativeArea(tree, heights, nbins):
    
    # traverse all the tree and compute the areas
    node = tree
    stack = []
    stack.append(node)
    while len(stack) > 0:
        currentNode = stack.pop()
        nodeParent = currentNode.parent

        if nodeParent != None:
            currentNode.areaAccumulated = currentNode.area + nodeParent.areaAccumulated
        
        '''
        print("    ")
        print("    ")
        print("---------------")
        print(nodeParent)
        print(currentNode)
        print("---------------")
        print("    ")
        print("    ")
        '''
        
        # put children on the stack
        children = currentNode.children
        for i in range(len(children)):
            stack.append(children[len(children)-1 - i])

def detectLeaves(tree, leaves):
    # traverse all the tree and detect the leaves
    node = tree
    stack = []
    stack.append(node)
    while len(stack) > 0:
        currentNode = stack.pop()
        nodeParent = currentNode.parent

        if nodeParent != None:
            currentNode.areaAccumulated = currentNode.area + nodeParent.areaAccumulated
        
        
        # check if the children is a leaf, in case -> put it in the list
        children = currentNode.children
        if (len(children) == 0):
            leaves.append(currentNode)
        
        # put children on the stack
        for i in range(len(children)):
            stack.append(children[len(children)-1 - i])    
