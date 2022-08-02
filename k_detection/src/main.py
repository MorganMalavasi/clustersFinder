import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
import cclustering_cpu as cc
import data_generation
import data_plot
import smoothing_detection, utility, histogram_clustering_hierarchical
from analysis.analysis_alg import internal_analysis


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
# remove 10% of the noise in the data
maxHeight = max(hist)
maxHeight_5_percent = maxHeight / 20 
for i in range(hist.shape[0]):
    if hist[i] < maxHeight_5_percent:
        hist[i] = 0

data_plot.plot_scatter(hist, bins, mode=2)
data_plot.plot_hist(hist, bins)

'''
# smoothing 
# smooth values with average of ten values
# we are interested in the hist values because they represent the values to divide
hist_smoothed_weighted = smoothing_detection.smooth_weighted(hist)
data_plot.plot_scatter(hist_smoothed_weighted, bins, mode=2)
data_plot.plot_hist(hist_smoothed_weighted, bins)
'''

# new algorithm for counting the number of clusters in an histogram of densities
clusters = histogram_clustering_hierarchical.getClustersFromHistogram(hist, bins)
thetaLabels = histogram_clustering_hierarchical.labelTheSamples(samples, theta, clusters, bins)

data_plot.plot_circle(theta, thetaLabels)





"""

# computing histogram values in 512 bins
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


"""

'''
K-MEANS
In K-means clustering, we don't know the correct number of clusters, so we make tests to see 
the one who suites better.
Here, we have the correct number, that we can get from the labels, so we will use it here.
'''

internalAnalysis = internal_analysis()
correctNumberOfClusters = max(labels) + 1

print("******************************************")
print("****** silhouette index*******************")
print("******************************************")

print("Computing silhouette in k-means knowing correct number of clusters")
internalAnalysis.k_means_silhouette_nrClusters_defined(correctNumberOfClusters, samples)
print("Computing silhouette in k-means without knowing correct number of clusters")
internalAnalysis.k_means_silhouette(samples)
print("Computing silhouette in circleClustering")
internalAnalysis.circleClustering_silhouette(samples, thetaLabels)

print("******************************************")
print("****** Calinski - Harabasz ***************")
print("******************************************")

print("Computing Calinski - Harabasz in k-means knowing correct number of clusters")
internalAnalysis.k_means_calinski_harabasz_nrClusters_defined(correctNumberOfClusters, samples)
print("Computing Calinski - Harabasz in k-means without knowing correct number of clusters")
internalAnalysis.k_means_calinski_harabasz(samples)
print("Computing Calinski - Harabasz in circleClustering")
internalAnalysis.circleClustering_calinski_harabasz(samples, thetaLabels)

print("******************************************")
print("****** Dunn Index ************************")
print("******************************************")

print("Computing Dunn Index in k-means knowing correct number of clusters")
internalAnalysis.k_means_dunn_nrClusters_defined(nr_clusters=correctNumberOfClusters, data=samples)
print("Computing Dunn Index in k-means without knowing correct number of clusters")
internalAnalysis.k_means_dunn(samples)
print("Computing Dunn Index in circleClustering")
internalAnalysis.circleClustering_dunn(samples, thetaLabels)