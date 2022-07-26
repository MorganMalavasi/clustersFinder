import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import cclustering_cpu as cc
import data_generation
import data_plot
import smoothing_detection, utility, histogram_clustering_hierarchical

plt.style.use('ggplot')

# constants
PI = np.pi
PI = np.float32(PI)

listOfDatasets = data_generation.createDatasets()

dataset = listOfDatasets[3]
samples, labels, n_dataset = dataset[0], dataset[1], dataset[2]


'''CIRCLE CLUSTERING'''
numberOfSamplesInTheDataset = samples.shape[0]
theta = 2 * PI * np.random.rand(numberOfSamplesInTheDataset)
print("Computing weights, S and C...")
matrixOfWeights, S, C = cc.computing_weights(samples, theta)
print("Computing the loop...")
theta = cc.loop(matrixOfWeights, theta, S, C, 0.001)

data_plot.doPCA(samples, labels, n_dataset)
data_plot.plot_circle(theta, labels)

hist, bins = utility.histogram(theta, nbins=128)

# Plot the histogram
data_plot.plot_scatter(hist, bins, mode=2)
data_plot.plot_hist(hist, bins)
# smoothing 
# smooth values with average of ten values
# we are interested in the hist values because they represent the values to divide
hist_smoothed_weighted = smoothing_detection.smooth_weighted(hist)
data_plot.plot_scatter(hist_smoothed_weighted, bins, mode=2)
data_plot.plot_hist(hist_smoothed_weighted, bins)

# new algorithm for counting the number of clusters in an histogram density base view
n = histogram_clustering_hierarchical.getClustersFromHistogram(hist, bins)






"""

# commputing histogram values in 512 bins
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




'''
K-MEANS
In K-means clustering, we don't know the correct number of clusters, so we make tests to see 
the one who suites better.
Here, we have the correct number, that we can get from the labels, so we will use it here.
'''
print("Computing clustering k-means with help")
kmeans_with_help = cluster.KMeans(n_clusters=max(labels)).fit(samples)
print("Computing clustering k-means without help")
kmeans_without_help = cluster.KMeans().fit(samples)


"""