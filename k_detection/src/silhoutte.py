from sklearn import datasets, metrics
from sklearn.cluster import KMeans
#
# Load IRIS dataset
#
iris = datasets.load_iris()
X = iris.data
y = iris.target

'''
print(X)
print("------------------------------------")
print(y)
print("------------------------------------")
print("------------------------------------")

for i in range(X.shape[0]):
    for k in range(X.shape[1]):
        print(X[i,k], end = ' ')
    print("Label = {0}".format(y[i]))

'''


#
# Instantiate the KMeans models
#
km = KMeans(n_clusters=3, random_state=42)
#
# Fit the KMeans model
#
km.fit_predict(X)
#
# Calculate Silhoutte Score
#
score = metrics.silhouette_score(X, km.labels_, metric='euclidean')
#
# Print the score
#
print('Silhouetter Score: %.3f' % score)