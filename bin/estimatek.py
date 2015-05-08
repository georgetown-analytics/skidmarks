# -*- coding: utf-8 -*-

###############################################################################
# Information
###############################################################################
# Created by Linwood Creekmore 

# From Mastering Machine Learning with scikit-learn, page 125

# In partial fulfillment of the requirements for the Georgetown University Data Analytics Graduate Certificate Program

# March 5, 2015

# https://plus.google.com/+LinwoodCreekmoreIII/

###############################################################################
# Imports
###############################################################################

import os
import numpy as np
import pandas as pd
from sklearn import metrics
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.preprocessing import Imputer
from statsmodels.sandbox.regression.predstd import wls_prediction_std

# Some colors for later
colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)


###############################################################################
# File Path Info
###############################################################################


path = path = os.path.abspath(os.getcwd())

#load data from a CSV to a dataframe
with open(os.path.normpath(os.path.join(path,'lin.csv'))) as in_data:
    skid_data = pd.DataFrame.from_csv(in_data, sep=',')


###############################################################################
# Load Data and Prep for scikit-learn
###############################################################################
train = len(skid_data) * .60
test = int(train + 1)
as_array = np.asfarray(skid_data[:int(train)][['Deceleration Events','Max Acceleration (mph per s)']]) #'Velocity Stdev','Average Acceleration (mph per s)', 'Max Acceleration (mph per s)', ' Acceleration Stdev','Displacement','Total Distance Traveled','Max Direction Change per sec', ' Direction Stdev','Time (s)', 'Turns', 'Aggressive Turns', 'Stops', 'Large Deceleration Events', 'Deceleration Events', 'Max Deceleration Event']])

###############################################################################
# scikit-learn preprocessing; uncomment to see differences
###############################################################################

#Correct missing data 
imputer = Imputer(missing_values="NaN", strategy="mean")
patched = imputer.fit_transform(as_array)

# Preprocessing tricks
#patched = StandardScaler().fit_transform(patched)
#patched = scale(patched, axis=0, with_mean=True)
#patched = preprocessing.normalize(patched, norm='l2')
#min_max_scaler = preprocessing.MinMaxScaler()
#patched = min_max_scaler.fit_transform(patched)

###############################################################################
# Main Functions
###############################################################################

K = range(1,20)
meandisortions = []

for k in K:
	kmeans = KMeans(n_clusters = k)
	kmeans.fit(patched)
	meandisortions.append(sum(np.min(cdist(patched,kmeans.cluster_centers_,'euclidean'),axis=1))/patched.shape[0])

plt.plot(K,meandisortions,'bx-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Average distortion')
plt.title('Selecting k with the Elbow Method')
plt.show()

###############################################################################
# Run the KMeans Cluster based on the Elbow Plot
###############################################################################

n_clusters = int(raw_input("Enter your k based on the elbow plot:\n>"))
cluster = KMeans(n_clusters=n_clusters, n_init = 20)
cluster.fit_transform(patched)
labels = cluster.labels_

# Evaluate the cluster using the Silhouette Coefficient Score
SilhouetteCoefficient = metrics.silhouette_score(patched, labels, metric='euclidean')
print "The Silhouette Coefficient is", SilhouetteCoefficient

# Find centers
centers = cluster.cluster_centers_
center_colors = colors[:len(centers)]
plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)

#plt.subplot(1,4,idx+1)
plt.scatter(patched[:, 0], patched[:, 1], color=colors[labels].tolist(), s=10)

plt.xticks(())
plt.yticks(())
plt.ylabel('Feature space for the Deceleration feature')
plt.xlabel('Feature space for the Max Acceleration feature')
plt.show()

###############################################################################
# Ordinary Least Squares Report
###############################################################################

model = sm.OLS(labels, patched)
results = model.fit()
print results.summary()



