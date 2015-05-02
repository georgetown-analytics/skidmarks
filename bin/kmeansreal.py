# -*- coding: utf-8 -*-

###############################################################################
# Information
###############################################################################
# Created by Linwood Creekmore 
# Input by Vikkram Mittal


# In partial fulfillment of the requirements for the Georgetown University Data Analytics Graduate Certificate Program

# April 19, 2015

# https://plus.google.com/+LinwoodCreekmoreIII/

###############################################################################
# Imports
###############################################################################

print(__doc__)

import os
import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split



###############################################################################
# File Paths
###############################################################################

path = os.path.abspath(os.getcwd())

###############################################################################
# Main Functions
###############################################################################

# Load my data in
with open("./lin.csv") as in_data:
    skid_data = pd.DataFrame.from_csv(in_data, sep=',')
as_array = np.asfarray(skid_data[['Average Velocity (mph)','Max Velocity', 'Velocity Stdev','Average Acceleration (mph per s)', 'Max Acceleration (mph per s)', ' Acceleration Stdev','Displacement','Total Distance Traveled','Max Direction Change per sec', ' Direction Stdev','Time (s)', 'Turns', 'Aggressive Turns', 'Stops', 'Large Deceleration Events', 'Deceleration Events', 'Max Deceleration Event']])

#number of groups
n_clusters=5

n_samples, n_features = skid_data.shape
n_digits = (len(as_array))


#Correct missing data 


# Preprocessing the data

imputer = Imputer(missing_values="NaN", strategy="mean")
patched = imputer.fit_transform(as_array)

cluster = KMeans(init='k-means++', n_clusters=n_clusters, n_init=20)

#patched = StandardScaler().fit_transform(patched)
#patched = scale(patched, axis=0, with_mean=True)

patched_normalized = preprocessing.normalize(patched, norm='l2')

#min_max_scaler = preprocessing.MinMaxScaler()
#patched_minmax = min_max_scaler.fit_transform(patched)

sample_size = len(skid_data)

labels = cluster.predict(patched_normalized)

print("n_clusters: %d, \t n_samples %d, \t n_features %d"
      % (n_clusters, n_samples, n_features))


print(79 * '_')
print('% 9s' % 'init'
      '    time  inertia    homo   compl  v-meas     ARI AMI  silhouette')


def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(patched_normalized)
    print('% 9s   %.2fs    %i   %.3f '
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))

bench_k_means(KMeans(init='k-means++', n_clusters=n_clusters, n_init=10),
              name="k-means++", data=patched_normalized)

bench_k_means(KMeans(init='random', n_clusters=n_clusters, n_init=10),
              name="random", data=patched_normalized)

# in this case the seeding of the centers is deterministic, hence we run the
# kmeans algorithm only once with n_init=1
pca = PCA(n_components=n_features).fit(patched_normalized)
bench_k_means(KMeans(init=pca.components_, n_clusters=n_clusters, n_init=1),
              name="PCA-based",
              data=patched_normalized)
print(79 * '_')

###############################################################################
# Visualize the results on PCA-reduced data

reduced_data = PCA(n_components=2).fit_transform(patched_normalized)
kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() + 1, reduced_data[:, 0].max() - 1
y_min, y_max = reduced_data[:, 1].min() + 1, reduced_data[:, 1].max() - 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()