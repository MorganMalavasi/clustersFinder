import numpy as np
import cclustering_cpu as cc
import data_generation
import data_plot

# constants
PI = np.pi
PI = np.float32(PI)

listOfDatasets = data_generation.createDatasets()

dataset = listOfDatasets[0]
samples, labels, n_dataset = dataset[0], dataset[1], dataset[2]


# get the theta 
numberOfSamplesInTheDataset = samples.shape[0]
theta = 2 * PI * np.random.rand(numberOfSamplesInTheDataset)
print("Computing weights, S and C...")
matrixOfWeights, S, C = cc.computing_weights(samples, theta)
print("Computing the loop...")
theta = cc.loop(matrixOfWeights, theta, S, C, 0.001)

'''
# plot the data
data_plot.doPCA(samples, labels, n_dataset)
data_plot.plot_circle(theta_CPU, eachTuple[1])
hist, bins = data_plot.histogram(theta_CPU, nbins=256)
data_plot.plot_hist(hist, bins, mode=0)
'''

