import os
import numpy as np
import matplotlib.pyplot as plt
import cclustering_cpu as cc
import data_generation
import data_plot
import smoothing_detection, utility, histogram_clustering_hierarchical
from analysis.analysis_alg import internal_analysis
from utility import averageOfList
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from multiprocessing import Process

os.environ["KMP_WARNINGS"] = "FALSE" 

plt.style.use('ggplot')
console = Console()

# constants
PI = np.pi
PI = np.float32(PI)

M = 6   # number of datasets
N = 3  # number of times for repeating our analysis


if __name__ == '__main__':  # TODO -> render multiprocess 
    for x in range(0, M):

        console.print("DATASET -> {0}".format(x+1))

        # kmeans
        silhoutteKmeans = []
        calinski_harabaszKmeans = []
        dunnKmeans = []
        pearsonKmeans = []
        avgKmeans = []
        separation_indexKmeans = []
        entropyKmeans = []
        wwcgKmeans = []
        prediction_strengthKmeans = []
        cvnnKmeans = []

        # circle clustering
        silhoutteCircleClustering = []
        calinski_harabaszCircleClustering = []
        dunnCircleClustering = []
        pearsonCircleClustering = []
        avgCircleClustering = []
        separation_indexCircleClustering = []
        entropyCircleClustering = []
        wwcgCircleClustering = []
        prediction_strengthCircleClustering = []
        cvnnCircleClustering = []

        table = Table(title = "dataset {0}".format(x+1))
        table.add_column("index",style="cyan", no_wrap=True)
        table.add_column("kmeans",style="magenta", no_wrap=True)
        table.add_column("circleClustering",style="green", no_wrap=True)

        for j in tqdm(range(0, N), desc="computing"):
            
            dataset = data_generation.createDatasets(x)
            samples, labels, n_dataset = dataset[0], dataset[1], dataset[2]
            
            #  if j == 0:
                # console.print("samples = {0}, centroids = {1}".format(samples.shape[0], max(labels) + 1), )
            
            with console.status("[bold green]Computing indexes loop {0}...".format(j+1)) as status:
                
                '''CIRCLE CLUSTERING'''
                numberOfSamplesInTheDataset = samples.shape[0]
                theta = 2 * PI * np.random.rand(numberOfSamplesInTheDataset)
                matrixOfWeights, S, C = cc.computing_weights(samples, theta)
                theta = cc.loop(matrixOfWeights, theta, S, C, 0.001)

                # data_plot.doPCA(samples, labels, n_dataset)
                # data_plot.plot_circle(theta)
                hist, bins = utility.histogram(theta, nbins=128)
                # Plot the histogram
                # data_plot.plot_scatter(hist, bins, mode=2)
                # data_plot.plot_hist(hist, bins)
                # remove 10% of the noise in the data
                maxHeight = max(hist)
                maxHeight_5_percent = maxHeight / 20 
                for i in range(hist.shape[0]):
                    if hist[i] < maxHeight_5_percent:
                        hist[i] = 0
                # data_plot.plot_scatter(hist, bins, mode=2)
                # data_plot.plot_hist(hist, bins)

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
                centroids = histogram_clustering_hierarchical.centroidsFinder(samples, thetaLabels)
                # data_plot.plot_circle(theta, thetaLabels)

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
                #print("there are {0} clusters".format(nClusters))

                # 2) smoothed
                nClusters, weights = smoothing_detection.simple_detection(hist_smoothed)
                #print("there are {0} smoothed clusters".format(nClusters))

                # 2) smoothed with weights
                nClusters, weights = smoothing_detection.simple_detection(hist_smoothed_weighted)
                #print("there are {0} smoothed clusters with weights".format(nClusters))


                """

                internalAnalysis = internal_analysis()
                correctNumberOfClusters = max(labels) + 1
                
                #print("**********************************************************************")
                #print("****** silhouette index **********************************************")
                #print("**********************************************************************")
        
                #print("Computing silhouette in k-means knowing correct number of clusters")
                silhoutteKmeans.append(internalAnalysis.k_means_silhouette_nrClusters_defined(correctNumberOfClusters, samples))
                #print("Computing silhouette in k-means without knowing correct number of clusters")
                # internalAnalysis.k_means_silhouette(samples)
                #print("Computing silhouette in circleClustering")
                silhoutteCircleClustering.append(internalAnalysis.circleClustering_silhouette(samples, thetaLabels))

                #print("**********************************************************************")
                #print("****** Calinski - Harabasz *******************************************")
                #print("**********************************************************************")

                #print("Computing Calinski - Harabasz in k-means knowing correct number of clusters")
                calinski_harabaszKmeans.append(internalAnalysis.k_means_calinski_harabasz_nrClusters_defined(correctNumberOfClusters, samples))
                #print("Computing Calinski - Harabasz in k-means without knowing correct number of clusters")
                # internalAnalysis.k_means_calinski_harabasz(samples)
                #print("Computing Calinski - Harabasz in circleClustering")
                calinski_harabaszCircleClustering.append(internalAnalysis.circleClustering_calinski_harabasz(samples, thetaLabels))

                #print("**********************************************************************")
                #print("****** Dunn Index ****************************************************")
                #print("**********************************************************************")

                #print("Computing Dunn Index in k-means knowing correct number of clusters")
                dunnKmeans.append(internalAnalysis.k_means_dunn_nrClusters_defined(nr_clusters=correctNumberOfClusters, data=samples))
                #print("Computing Dunn Index in k-means without knowing correct number of clusters")
                # internalAnalysis.k_means_dunn(samples)
                #print("Computing Dunn Index in circleClustering")
                dunnCircleClustering.append(internalAnalysis.circleClustering_dunn(samples, thetaLabels))

                #print("**********************************************************************")
                #print("****** Pearson Index ****************************************************")
                #print("**********************************************************************")

                #print("Computing pearson Index in k-means knowing correct number of clusters")
                pearsonKmeans.append(internalAnalysis.k_means_pearson_nrClusters_defined(correctNumberOfClusters, samples))
                #print("Computing pearson Index in k-means without knowing correct number of clusters")
                # internalAnalysis.k_means_pearson(samples)
                #print("Computing pearson Index in circleClustering")
                pearsonCircleClustering.append(internalAnalysis.circleClustering_pearson(samples, thetaLabels))

                #print("**********************************************************************")
                #print("****** Average within-cluster dissimilarities ************************")
                #print("**********************************************************************")

                #print("Computing Average within-cluster dissimilarities in k-means knowing correct number of clusters")
                avgKmeans.append(internalAnalysis.k_means_average_within_cluster_dissimilarities_nrClusters_defined(nr_clusters = correctNumberOfClusters, data = samples))
                #print("Computing Average within-cluster dissimilarities in k-means without knowing correct number of clusters")
                # internalAnalysis.k_means_average_within_cluster_dissimilarities(samples)
                #print("Computing Average within-cluster dissimilarities in circleClustering")
                avgCircleClustering.append(internalAnalysis.circleClustering_average_within_cluster_dissimilarities(samples, thetaLabels))

                #print("**********************************************************************")
                #print("****** Separation Index **********************************************")
                #print("**********************************************************************")

                #print("Computing Separation Index in k-means knowing correct number of clusters")
                separation_indexKmeans.append(internalAnalysis.k_means_separation_index_nrClusters_defined(nr_clusters = correctNumberOfClusters, data = samples))
                #print("Computing Separation Index in k-means without knowing correct number of clusters")
                # internalAnalysis.k_means_separation_index(samples)
                #print("Computing Separation Index in circleClustering")
                separation_indexCircleClustering.append(internalAnalysis.circleClustering_separation_index(samples, thetaLabels))

                #print("**********************************************************************")
                #print("****** Entropy *******************************************************")
                #print("**********************************************************************")

                #print("Computing Uniformity of cluster sizes in k-means knowing correct number of clusters")
                entropyKmeans.append(internalAnalysis.k_means_entropy_nrClusters_defined(correctNumberOfClusters, samples))
                #print("Computing Uniformity of cluster sizes in k-means without knowing correct number of clusters")
                # internalAnalysis.k_means_entropy(samples)
                #print("Computing Uniformity of cluster sizes in circleClustering")
                entropyCircleClustering.append(internalAnalysis.circleClustering_entropy(samples, thetaLabels))

                #print("**********************************************************************")
                #print("****** widest within-cluster gap *************************************")
                #print("**********************************************************************")

                #print("Computing widest within cluster gap in k-means knowing correct number of clusters")
                wwcgKmeans.append(internalAnalysis.k_means_wwcg_nrClusters_defined(correctNumberOfClusters, samples))
                #print("Computing  widest within cluster gap in k-means without knowing correct number of clusters")
                # internalAnalysis.k_means_wwcg(samples)
                #print("Computing  widest within cluster gap in circleClustering")
                wwcgCircleClustering.append(internalAnalysis.circleClustering_wwcg(samples, thetaLabels))

                #print("**********************************************************************")
                #print("****** prediction strength (PS) **************************************")
                #print("**********************************************************************")

                #print("Computing prediction strength in k-means knowing correct number of clusters")
                # prediction_strengthKmeans.add(internalAnalysis.k_means_predictionStrength_nrClusters_defined(correctNumberOfClusters, samples))
                #print("Computing prediction strength in k-means without knowing correct number of clusters")
                # internalAnalysis.k_means_predictionStrength(samples)
                #print("Computing prediction strength in circleClustering")
                # internalAnalysis.circleClustering_predictionStrength(samples, thetaLabels)

                #print("**********************************************************************")
                #print("****** clustering validity index based on Nearest Neighbours *********")
                #print("**********************************************************************")

                #print("Computing cvnn in k-means knowing correct number of clusters")
                score_cvnn_kmeans, score_cvnn_circleClustering = internalAnalysis.k_means_cvnn_nrClusters_defined(correctNumberOfClusters, samples, thetaLabels)
                cvnnKmeans.append(score_cvnn_kmeans)
                cvnnCircleClustering.append(score_cvnn_circleClustering)

                # console.log(f"[green]Finish computing data loop = [/green] {j+1}")

        '''
        # kmeans
        silhoutteKmeans = []
        calinski_harabaszKmeans = []
        dunnKmeans = []
        pearsonKmeans = []
        avgKmeans = []
        separation_indexKmeans = []
        entropyKmeans = []
        wwcgKmeans = []
        prediction_strengthKmeans = []
        cvnnKmeans = []

        # circle clustering
        silhoutteCircleClustering = [] 
        calinski_harabaszCircleClustering = []
        dunnCircleClustering = []
        pearsonCircleClustering = []
        avgCircleClustering = []
        separation_indexCircleClustering = []
        entropyCircleClustering = []
        wwcgCircleClustering = []
        prediction_strengthCircleClustering = []
        cvnnCircleClustering = []
        
        '''
        str(averageOfList(silhoutteKmeans))
        str(averageOfList(calinski_harabaszKmeans))
        table.add_row("silhoutte", str(averageOfList(silhoutteKmeans)), str(averageOfList(silhoutteCircleClustering)))
        table.add_row("calinski_harabasz", str(averageOfList(calinski_harabaszKmeans)), str(averageOfList(calinski_harabaszCircleClustering)))
        table.add_row("dunn", str(averageOfList(dunnKmeans)), str(averageOfList(dunnCircleClustering)))
        table.add_row("pearson", str(averageOfList(pearsonKmeans)), str(averageOfList(pearsonCircleClustering)))
        table.add_row("avg", str(averageOfList(avgKmeans)), str(averageOfList(avgCircleClustering)))
        table.add_row("separation index", str(averageOfList(separation_indexKmeans)), str(averageOfList(separation_indexCircleClustering)))
        table.add_row("entropy", str(averageOfList(entropyKmeans)), str(averageOfList(entropyCircleClustering)))
        table.add_row("wwcg", str(averageOfList(wwcgKmeans)), str(averageOfList(wwcgCircleClustering)))
        table.add_row("cvnn", str(averageOfList(cvnnKmeans)), str(averageOfList(cvnnCircleClustering)))

        print()
        console.print(table)

        # print(cvnnKmeans)
        # print(cvnnCircleClustering)