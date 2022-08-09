import os
import numpy as np
import matplotlib.pyplot as plt
from analysis.utils import clusterFinder, createFile
from sklearn.metrics.pairwise import euclidean_distances
import networkx as nx
import pandas as pd
import subprocess

def average(list_):
    sum = 0.0
    for i in range(len(list_)):
        sum += list_[i]
    return sum / len(list_)

def minimum_spanning_tree(samples, labels):
    
    averagePaths = []

    for c in range(int(max(labels) + 1)):
        cluster = clusterFinder(c, labels, samples)
        cluster_numpy = np.asarray(cluster)

        matrixOfDissimilarities = euclidean_distances(cluster_numpy, cluster_numpy)
        G = nx.Graph()
        for nodeStart in range(cluster_numpy.shape[0]):
            for nodeEnd in range(nodeStart + 1, cluster_numpy.shape[0]):
                G.add_edge(nodeStart, nodeEnd, weight=matrixOfDissimilarities[nodeStart, nodeEnd], node_color="g")

        mst = nx.minimum_spanning_tree(G, weight='weight', algorithm='kruskal')

        '''
        node_color_list = [nc for _, nc in G.nodes(data="node_color")]
        pos = nx.spectral_layout(G)
        plt.figure(figsize=(8, 8))
        nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color="k", )
        nx.draw_networkx_nodes(G, pos, alpha=0.8, node_color=node_color_list)
        nx.draw_networkx_labels(G, pos, font_size=14)
        plt.axis("off")
        plt.title("The original graph.")
        plt.show()

        node_color_list = [nc for _, nc in mst.nodes(data="node_color")]
        pos = nx.spectral_layout(mst)
        plt.figure(figsize=(8, 8))
        nx.draw_networkx_edges(mst, pos, alpha=0.3, edge_color="k")
        nx.draw_networkx_nodes(mst, pos, alpha=0.8, node_color=node_color_list)
        nx.draw_networkx_labels(mst, pos, font_size=14)
        plt.axis("off")
        plt.title("The minimum spanning tree")
        plt.show()
        '''
        maximumPaths = []
        for nodeStart in range(cluster_numpy.shape[0]):
            for nodeEnd in range(nodeStart + 1, cluster_numpy.shape[0]):
                paths_generator = nx.all_simple_paths(mst, nodeStart, nodeEnd)
                paths = list(paths_generator)
                paths = paths[0]
                # print(paths)

                '''
                # there is only one path for each couple of nodes, because we are dealing 
                # with a minimum spanning tree
                maximumPaths.append(max(nx.all_simple_paths(mst, nodeStart, nodeEnd), key=lambda x: len(x)))
                print("--------- from = {0} , to = {1}".format(nodeStart, nodeEnd))
                for eachPath in paths:
                    print(eachPath)
                '''
                # print("from = {0}, to = {1}".format(nodeStart, nodeEnd))
                sum = 0.0
                for i in range(len(paths) - 1):
                    elem_1 = paths[i]
                    elem_2 = paths[i + 1]
                    # print("x = {0}, y = {1}, weight = {2}".format(elem_1, elem_2, mst[elem_1][elem_2]["weight"]))
                    sum += mst[elem_1][elem_2]["weight"]
                maximumPaths.append(sum)

        averagePaths.append(max(maximumPaths))

    return average(averagePaths)

def wwcg(samples, labels):
    # Defining the R script and loading the instance in Python
    createFile(samples, labels)
    test()

    return 

def test():
    command = 'Rscript'
    # command = 'Rscript'                    # OR WITH bin FOLDER IN PATH ENV VAR 
    arg = '--vanilla' 

    try: 
        p = subprocess.Popen([command, arg,
                            "analysis/cqcluster/widest_within_cluster_gap.R"],
                            cwd = os.getcwd(),
                            stdin = subprocess.PIPE, 
                            stdout = subprocess.PIPE, 
                            stderr = subprocess.PIPE) 

        output, error = p.communicate() 

        if p.returncode == 0: 
            print('R OUTPUT:\n {0}'.format(output.decode("utf-8"))) 
            print()
        else: 
            print('R ERROR:\n {0}'.format(error.decode("utf-8"))) 

        return True

    except Exception as e: 
        print("dbc2csv - Error converting file: ") 
        print(e)

        return False