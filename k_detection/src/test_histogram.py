import numpy as np
import data_plot
import histogram_clustering_hierarchical

height = np.array([1, 2, 4, 15, 8, 7, 5, 19, 12, 19, 5, 4, 2, 5, 1])
bins = np.array(range(0,15))

print(height)
print(bins)

# data_plot.plot_hist(height, bins)
n = histogram_clustering_hierarchical.getClustersFromHistogram(height, bins)
print(n)