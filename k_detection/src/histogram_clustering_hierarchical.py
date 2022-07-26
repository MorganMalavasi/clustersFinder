import numpy as np
from anytree import AnyNode, RenderTree
from anytree.exporter import DotExporter

class Tree:
    def __init__(self, name='root', children=None, nbins = 512, weight = 0.0):
        self.name = name
        self.children = []
        self.interval = (0.0, nbins)
        self.weight = weight
        if children is not None:
            for child in children:
                self.add_child(child)
    def __repr__(self):
        return self.name
    def add_child(self, node, interval, weight):
        assert isinstance(node, Tree)
        self.children.append(node, interval, weight)
        

def getClustersFromHistogram(heights, nbins):
    numberOfClusters = 0

    # start alg
    tree = AnyNode(id="lv0_0", interval = (0.0, nbins.shape[0]), area = 0.0)
    createHierarchicalTree(tree, heights, nbins)
    # print(RenderTree(tree))
    DotExporter(tree).to_dotfile("tree.dot")
    return numberOfClusters



# return the correct parent between all the parents 
def correctParent(parents, intervalOfNode, maxLength):
    correctParent = None

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
    id = _parent_.id
    newLv = int(id[2:3]) + 1
    n_children = len(_parent_.children)
    newId = "lv" + str(newLv) + "_" + str(n_children)
    newNode = AnyNode(id=newId, interval = intervalChild, area = areaChild, parent=_parent_)
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
    