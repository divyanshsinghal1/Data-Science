from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
data = iris.data
# Create a KMeans object with number of clusters
kmeans = KMeans(n_clusters=3)

# Fit the KMeans object to the data
kmeans.fit(data)

# Predict the cluster labels for the data
labels = kmeans.predict(data)

# Compute the silhouette score for the clustering
silhouette_avg = silhouette_score(data, labels)

# Get the sum of squared distances for the clusters
inertia = kmeans.inertia_

#Finding optimal clusters using Elbow method

# Compute the sum of squared distances for different number of clusters
ssd = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    ssd.append(kmeans.inertia_)
    
# Plot the sum of squared distances against the number of clusters
plt.plot(range(1, 11), ssd, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of squared distances')
plt.title('Elbow Method')
plt.show()

#Finding optimal clusters using Silhouette analysis


# Compute the silhouette score for different number of clusters
silhouette_avgs = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(data)
    silhouette_avg = silhouette_score(data, labels)
    silhouette_avgs.append(silhouette_avg)
    
# Plot the silhouette score against the number of clusters
plt.plot(range(2, 11), silhouette_avgs, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette score')
plt.title('Silhouette Analysis')
plt.show()

#Visualizing clusters using scatter plot

# Create a scatter plot of the data colored by cluster labels
kmeans = KMeans(n_clusters=3)
kmeans.fit(data)
labels = kmeans.predict(data)
plt.scatter(data[:, 0], data[:, 1], c=labels)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-means Clustering')
plt.show()

#Visualizing clusters using heatmap


# Add the cluster labels as a column to the data
data_labeled = np.hstack((data, labels.reshape((-1, 1))))

# Create a heatmap of the data points colored by cluster labels
sns.heatmap(data_labeled, cmap='coolwarm')
plt.xlabel('Features')
plt.ylabel('Data points')
plt.title('K-means Clustering Heatmap')
plt.show()

#Optimizing clusters using feature scaling


# Scale the input features to have zero mean and unit variance
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Perform K-means clustering on the scaled data
kmeans = KMeans(n_clusters=3)
kmeans.fit(data_scaled)
labels = kmeans.predict(data_scaled)

#Optimizing clusters using feature selection


# Select the top k features with highest F-score
selector = SelectKBest(f_regression, k=2)
data_selected = selector.fit_transform(data, labels)

#Optimizing clusters using different distance metrics

# Define the distance metrics to use
metrics = ['euclidean', 'manhattan', 'cosine']

# Perform K-means clustering with different distance metrics
for metric in metrics:
    kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, random_state=0)
    kmeans.fit(data)
    labels = kmeans.predict(data)
    centroids = kmeans.cluster_centers_
    closest, _ = pairwise_distances_argmin_min(centroids, data, metric=metric)
    
    # Visualize the clustering with the chosen metric
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=300, c='black')
    plt.scatter(data[closest, 0], data[closest, 1], marker='o', s=100, edgecolor='k', facecolor='none')
    plt.title(f'K-means Clustering with {metric} metric')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

