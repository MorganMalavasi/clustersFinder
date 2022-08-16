from cProfile import label
import sys
from analysis.base_dunn import dunn_fast
from analysis.pearson import pearson_index, pearson_index_R
from analysis.average_within_cluster_dissimilarities import average_within_cluster_dissimilarities
from analysis.separation_index import separationindex
from analysis.uniformity_of_cluster_size import entropy
from analysis.widest_within_cluster_gap import minimum_spanning_tree_max_distance, widest_within_cluster_gap_formula
from analysis.prediction_strength import predictionStrength
from analysis.nearest_neighbours import cvnn_formula
from sklearn import metrics
from sklearn.cluster import KMeans


class internal_analysis:
    
    minSizeCluster = 2 # min number of clusters
    maxSizeCluster = 15 # max number of clusters 
    size = maxSizeCluster - minSizeCluster 
    kMeans_ = None
    kMeans_list = None   

    def k_means_nrClusters(self, numberK, samples):
        return KMeans(n_clusters=numberK).fit(samples)

    def k_means_array(self, data):
        self.kMeans_list = []
        for i in range(self.minSizeCluster, self.maxSizeCluster):
            self.kMeans_list.append(self.k_means_nrClusters(numberK=i, samples=data))
        
    # ************************* silhouette ************************************
    def k_means_silhouette_nrClusters_defined(self, nr_clusters, data):
        if self.kMeans_ == None:
            self.kMeans_ = self.k_means_nrClusters(numberK = nr_clusters, samples = data)
        # compute silhoutte score 
        score = metrics.silhouette_score(data, self.kMeans_.labels_, metric="euclidean")
        print(score)

    def k_means_silhouette(self, data):
        maxScore = -(sys.maxsize)
        clusters = -1
        
        if self.kMeans_list == None:
            self.k_means_array(data)

        for i in range(len(self.kMeans_list)):
            score = metrics.silhouette_score(data, self.kMeans_list[i].labels_, metric="euclidean")
        
            if score > maxScore:
                maxScore = score
                clusters = max(self.kMeans_list[i].labels_) + 1
        
        print("score = {0}, nr clusters = {1}".format(maxScore, clusters))
                
    def circleClustering_silhouette(self, data, labels):
        score = metrics.silhouette_score(data, labels, metric="euclidean")
        print(score)

    # ************************* ch_calinski_harabasz **************************
    def k_means_calinski_harabasz_nrClusters_defined(self, nr_clusters, data):
        if self.kMeans_ == None:
            self.kMeans_ = self.k_means_nrClusters(numberK = nr_clusters, samples = data)
        # compute silhoutte score 
        score = metrics.calinski_harabasz_score(data, self.kMeans_.labels_)
        print(score)

    def k_means_calinski_harabasz(self, data):
        maxScore = -(sys.maxsize)
        clusters = -1
        
        if self.kMeans_list == None:
            self.k_means_array(data)

        for i in range(len(self.kMeans_list)):
            score = metrics.calinski_harabasz_score(data, self.kMeans_list[i].labels_)
        
            if score > maxScore:
                maxScore = score
                clusters = max(self.kMeans_list[i].labels_) + 1
        
        print("score = {0}, nr clusters = {1}".format(maxScore, clusters))
                
    def circleClustering_calinski_harabasz(self, data, labels):
        score = metrics.calinski_harabasz_score(data, labels)
        print(score)

    
    # ************************* dunn index ************************************
    def k_means_dunn_nrClusters_defined(self, nr_clusters, data):
        '''
        SLOW VERSION 

        # K-Means
        df = pd.DataFrame(data)
        k_means_ = KMeans(n_clusters=nr_clusters)
        k_means_.fit(df)
        y_pred = k_means_.predict(df)

        prediction = pd.concat([df, pd.DataFrame(y_pred, columns=['pred'])], axis = 1)

        k_list = []
        for i in range(nr_clusters):
            clus = prediction.loc[prediction.pred == i]
            k_list.append(clus.values)

        print(dunn_fast(k_list))
        '''

        if self.kMeans_ == None:
            self.kMeans_ = self.k_means_nrClusters(numberK = nr_clusters, samples = data)
        # compute silhoutte score 
        score = dunn_fast(points = data, labels = self.kMeans_.labels_)
        print(score)
        
    def k_means_dunn(self, data):
        maxScore = -(sys.maxsize)
        clusters = -1
        
        if self.kMeans_list == None:
            self.k_means_array(data)

        for i in range(len(self.kMeans_list)):
            score = dunn_fast(points = data, labels = self.kMeans_list[i].labels_)
        
            if score > maxScore:
                maxScore = score
                clusters = max(self.kMeans_list[i].labels_) + 1
        
        print("score = {0}, nr clusters = {1}".format(maxScore, clusters))
                
    def circleClustering_dunn(self, data, labels_):
        score = dunn_fast(points = data, labels = labels_)
        print(score)

    # ************************* clustering validity index based on Nearest Neighbours 
    def k_means_cvnn_nrClusters_defined(self, nr_clusters, data, labelsTheta):
        if self.kMeans_ == None:
            self.kMeans_ = self.k_means_nrClusters(numberK = nr_clusters, samples = data)
        # compute silhoutte score 
        # score = minimum_spanning_tree(data, self.kMeans_.labels_)
        scoreKmeans, scoreCircleClustering = cvnn_formula(data, self.kMeans_.labels_, labelsTheta)
        print("score of kmeans = {0}".format(scoreKmeans))
        print("score of circleClustering = {0}".format(scoreCircleClustering))

    # ************************* pearson index ************************************
    def k_means_pearson_nrClusters_defined(self, nr_clusters, data):
        if self.kMeans_ == None:
            self.kMeans_ = self.k_means_nrClusters(numberK = nr_clusters, samples = data)
        # compute silhoutte score 
        score = pearson_index_R(data, self.kMeans_.labels_)
        print(score)
        
    def k_means_pearson(self, data):
        maxScore = -(sys.maxsize)
        clusters = -1
        
        if self.kMeans_list == None:
            self.k_means_array(data)

        for i in range(len(self.kMeans_list)):
            score = pearson_index_R(data, self.kMeans_list[i].labels_)
        
            if score > maxScore:
                maxScore = score
                clusters = max(self.kMeans_list[i].labels_) + 1
        
        print("score = {0}, nr clusters = {1}".format(maxScore, clusters))
                
    def circleClustering_pearson(self, data, labels_):
        score = pearson_index_R(data, labels_)
        print(score)

    # ************************* average within-cluster dissimilarities **********
    def k_means_average_within_cluster_dissimilarities_nrClusters_defined(self, nr_clusters, data):
        if self.kMeans_ == None:
            self.kMeans_ = self.k_means_nrClusters(numberK = nr_clusters, samples = data)
        # compute silhoutte score 
        score = average_within_cluster_dissimilarities(data, self.kMeans_.labels_)
        print(score)
        
    def k_means_average_within_cluster_dissimilarities(self, data):
        maxScore = sys.maxsize
        clusters = -1
        
        if self.kMeans_list == None:
            self.k_means_array(data)

        for i in range(len(self.kMeans_list)):
            score = average_within_cluster_dissimilarities(data, self.kMeans_list[i].labels_)
        
            if score < maxScore:
                maxScore = score
                clusters = max(self.kMeans_list[i].labels_) + 1
        
        print("score = {0}, nr clusters = {1}".format(maxScore, clusters))
                
    def circleClustering_average_within_cluster_dissimilarities(self, data, labels_):
        score = average_within_cluster_dissimilarities(data, labels_)
        print(score)

    # ************************* separation index *******************************
    def k_means_separation_index_nrClusters_defined(self, nr_clusters, data):
        if self.kMeans_ == None:
            self.kMeans_ = self.k_means_nrClusters(numberK = nr_clusters, samples = data)
        # compute silhoutte score 
        score = separationindex(data, self.kMeans_.labels_)
        print(score)

    def k_means_separation_index(self, data):
        maxScore = -(sys.maxsize)
        clusters = -1
        
        if self.kMeans_list == None:
            self.k_means_array(data)

        for i in range(len(self.kMeans_list)):
            score = separationindex(data, self.kMeans_list[i].labels_)
        
            if score > maxScore:
                maxScore = score
                clusters = max(self.kMeans_list[i].labels_) + 1
        
        print("score = {0}, nr clusters = {1}".format(maxScore, clusters))
                
    def circleClustering_separation_index(self, data, labels_):
        score = separationindex(data, labels_)
        print(score)

    # ************************* entropy ***************************************
    def k_means_entropy_nrClusters_defined(self, nr_clusters, data):
        if self.kMeans_ == None:
            self.kMeans_ = self.k_means_nrClusters(numberK = nr_clusters, samples = data)
        # compute silhoutte score 
        score = entropy(data, self.kMeans_.labels_)
        print(score)

    def k_means_entropy(self, data):
        maxScore = -(sys.maxsize)
        clusters = -1
        
        if self.kMeans_list == None:
            self.k_means_array(data)

        for i in range(len(self.kMeans_list)):
            score = entropy(data, self.kMeans_list[i].labels_)
        
            if score > maxScore:
                maxScore = score
                clusters = max(self.kMeans_list[i].labels_) + 1
        
        print("score = {0}, nr clusters = {1}".format(maxScore, clusters))
                
    def circleClustering_entropy(self, data, labels_):
        score = entropy(data, labels_)
        print(score)

    # ************************* widest within cluster gap ********************
    def k_means_wwcg_nrClusters_defined(self, nr_clusters, data):
        if self.kMeans_ == None:
            self.kMeans_ = self.k_means_nrClusters(numberK = nr_clusters, samples = data)
        # compute silhoutte score 
        # score = minimum_spanning_tree(data, self.kMeans_.labels_)
        score = widest_within_cluster_gap_formula(data, self.kMeans_.labels_)
        print(score)

    def k_means_wwcg(self, data):
        minScore = (sys.maxsize)
        clusters = -1
        
        if self.kMeans_list == None:
            self.k_means_array(data)

        for i in range(len(self.kMeans_list)):
            score = widest_within_cluster_gap_formula(data, self.kMeans_list[i].labels_)
        
            if score < minScore:
                minScore = score
                clusters = max(self.kMeans_list[i].labels_) + 1
        
        print("score = {0}, nr clusters = {1}".format(minScore, clusters))
                
    def circleClustering_wwcg(self, data, labels_):
        score = widest_within_cluster_gap_formula(data, labels_)
        print(score)

    # ************************* prediction strength ***************************
    def k_means_predictionStrength_nrClusters_defined(self, nr_clusters, data):
        score = predictionStrength(data, nr_clusters)
        print(score)
    '''
    def k_means_predictionStrength(self, data):
        minScore = (sys.maxsize)
        clusters = -1
        
        if self.kMeans_list == None:
            self.k_means_array(data)

        for i in range(len(self.kMeans_list)):
            score = predictionStrength(data, self.kMeans_list[i].labels_)
        
            if score < minScore:
                minScore = score
                clusters = max(self.kMeans_list[i].labels_) + 1
        
        print("score = {0}, nr clusters = {1}".format(minScore, clusters))
                
    def circleClustering_predictionStrength(self, data, labels_):
        score = predictionStrength(data, labels_)
        print(score)

    '''