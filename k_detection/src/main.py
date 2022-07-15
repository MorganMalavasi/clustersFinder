import numpy as np
from sklearn import cluster
import cclustering_cpu as cc
import data_generation
import data_plot
import smoothing_detection

# constants
PI = np.pi
PI = np.float32(PI)

listOfDatasets = data_generation.createDatasets()

dataset = listOfDatasets[3]
samples, labels, n_dataset = dataset[0], dataset[1], dataset[2]


# get the theta 
numberOfSamplesInTheDataset = samples.shape[0]
theta = 2 * PI * np.random.rand(numberOfSamplesInTheDataset)
print("Computing weights, S and C...")
matrixOfWeights, S, C = cc.computing_weights(samples, theta)
print("Computing the loop...")
theta = cc.loop(matrixOfWeights, theta, S, C, 0.001)


# plot the data
data_plot.doPCA(samples, labels, n_dataset)
data_plot.plot_circle(theta, labels)
hist, bins = data_plot.histogram(theta, nbins=512)
data_plot.plot_hist(hist, bins, mode=2)

# smoothing 
# smooth values with average of ten values
# we are interested in the hist values because they represent the values to divide
hist_smoothed = smoothing_detection.smooth(hist)
data_plot.plot_hist(hist_smoothed, bins, mode=2)

hist_smoothed_weighted = smoothing_detection.smooth_weighted(hist)
data_plot.plot_hist(hist_smoothed_weighted, bins, mode=2)

# detection
# detect how many wells there are
# 1) in the real 
nClusters, weights = smoothing_detection.simple_detection(hist)
print("there are {0} clusters".format(nClusters))

# 2) smoothed
nClusters, weights = smoothing_detection.simple_detection(hist_smoothed)
print("there are {0} smoothed clusters".format(nClusters))

# 2) smoothed with weights
nClusters, weights = smoothing_detection.simple_detection(hist_smoothed_weighted)
print("there are {0} smoothed clusters with weights".format(nClusters))


