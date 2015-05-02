# -*- coding: utf-8 -*-

###############################################################################
# Information
###############################################################################
# Created by Linwood Creekmore 
# Input from Vikram Mittal

# In partial fulfillment of the requirements for the Georgetown University Data Analytics Graduate Certificate Program

# useful links
# http://stackoverflow.com/questions/27504870/sklearn-kmeans-get-class-centroid-labels-and-reference-to-a-dataset  -> returns the class of the cluster; can see where each driver's trip fell

# May 1, 2015

# https://plus.google.com/+LinwoodCreekmoreIII/

###############################################################################
# Imports
###############################################################################


print(__doc__)

from time import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import os
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.metrics.cluster import v_measure_score


path = path = os.path.abspath(os.getcwd())

# Some colors for later
colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)

'''
df = pd.read_csv('./lin.csv')

digits = np.asfarray(df[['driver_id', 'trip_id', 'Average Velocity (mph)', 'Max Velocity', 'Velocity Stdev','Average Acceleration (mph per s)', 'Max Acceleration (mph per s)', ' Acceleration Stdev','Displacement','Total Distance Traveled','Max Direction Change per sec', ' Direction Stdev','Time (s)', 'Turns', 'Aggressive Turns', 'Stops', 'Large Deceleration Events', 'Deceleration Events', 'Max Deceleration Event']])
data = scale(digits.data)



#####
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt

fig, axe = plt.subplots(figsize=(18, 4))
for idx, dataset in enumerate((circles, moons, blobs, noise)):
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    
'''  


###
 #load data from a CSV to a dataframe
with open("./lin.csv") as in_data:
    skid_data = pd.DataFrame.from_csv(in_data, sep=',')

n_samples, n_features = skid_data.shape
print skid_data.shape

#skid_data=skid_data.fillna(value=-999)

#load all numeric data into an array. The offense column from the crime data
#is excluded
as_array = np.asfarray(skid_data[['Average Velocity (mph)','Max Velocity', 'Velocity Stdev','Average Acceleration (mph per s)', 'Max Acceleration (mph per s)', ' Acceleration Stdev','Displacement','Total Distance Traveled','Max Direction Change per sec', ' Direction Stdev','Time (s)', 'Turns', 'Aggressive Turns', 'Stops', 'Large Deceleration Events', 'Deceleration Events', 'Max Deceleration Event']])

#number of groups
n_clusters=4


#Correct missing data 
imputer = Imputer(missing_values="NaN", strategy="mean")
patched = imputer.fit_transform(as_array)

# Preprocessing tricks
#patched = StandardScaler().fit_transform(patched)
#patched = scale(patched, axis=0, with_mean=True)

patched_normalized = preprocessing.normalize(patched, norm='l2')

#min_max_scaler = preprocessing.MinMaxScaler()
#patched_minmax = min_max_scaler.fit_transform(patched)




#cluster data 
cluster = KMeans(n_clusters=n_clusters)
cluster.fit(patched)



# assigned grouped labels to the Skid data
#labels = cluster.labels_
#skid_data["labels"]=labels
'''
  # Fit the model with our algorithm
cluster = MiniBatchKMeans(n_clusters=3)
cluster.fit(patched)
'''
reduced_data = PCA(n_components=2).fit_transform(patched)


kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=20)
kmeans.fit(reduced_data)

# Make Predictions
predictions = cluster.predict(patched)



# array of indexes corresponding to classes around centroids, in the order of your dataset
classified_data = kmeans.labels_
prediction_data = cluster.labels_

#copy dataframe (may be memory intensive but just for illustration)
skid_data = skid_data.copy()
#print pd.Series(classified_data)
#print pd.Series(prediction_data)
skid_data['Predicted Class'] = pd.Series(prediction_data, index=skid_data.index)
#print skid_data.describe()
print cluster.labels_
#print list(skid_data.columns)
skid_data.plot( x = 'Average Acceleration (mph per s)', y = 'Predicted Class', kind = 'scatter')
#plt.show()


# Scoring to evaluate cluster performance

# Silhouette Coefficient
print "We want scores close to 1 \n"

SilouetteCoefficient = metrics.silhouette_score(patched, classified_data, metric='euclidean')
AdjustRandIndex = metrics.adjusted_rand_score(classified_data, prediction_data)
MutualInfoScore = metrics.adjusted_mutual_info_score(classified_data,prediction_data)
HomogenietyScore = metrics.homogeneity_score(classified_data, prediction_data) 
CompletenessScore = metrics.completeness_score(classified_data, prediction_data)
V_measure = metrics.v_measure_score(classified_data, prediction_data) 


print "The Silouette Coefficient score is %r\nThe Adjusted Rand index is %r\nThe Mutual Information based score is %r\nThe Homogeneity score is %r\nThe completeness score is %r\nThe V-measure score is %r" % (SilouetteCoefficient,AdjustRandIndex,MutualInfoScore,HomogenietyScore,CompletenessScore,V_measure)




'''
#Ben Bengfort Visualization

# Find centers
centers = cluster.cluster_centers_
center_colors = colors[:len(centers)]
plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)

#plt.subplot(1,4,idx+1)
plt.scatter(patched[:, 0], patched[:, 1], color=colors[predictions].tolist(), s=10)

plt.xticks(())
plt.yticks(())
plt.ylabel('Some random values')
plt.xlabel('Some random units')

plt.show()
'''
#############
#scikit-learn visualization example

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
plt.title('K-means clustering on the snippet of Team Skidmarks dataset \n(PCA-reduced data)'
          'Centroids are marked with blue cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
#plt.savefig('5clusterPCA.png', orientation = 'landscape')
#plt.show()
figsavepath = os.path.normpath(os.path.join(path,'figures',str(n_clusters)+"_cluster_KMeans_PCAReduced"+ ".png"))


