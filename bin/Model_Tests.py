# -*- coding: utf-8 -*-

###############################################################################
# Information
###############################################################################
# Created by Linwood Creekmore 

# From Mastering Machine Learning with scikit-learn, page 125

# In partial fulfillment of the requirements for the Georgetown University Data Analytics Graduate Certificate Program

# May 12, 2015

# https://plus.google.com/+LinwoodCreekmoreIII/

import os
import numpy as np
import pandas as pd
from sklearn import metrics
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.preprocessing import Imputer
import time as time
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.cluster import AgglomerativeClustering


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

X = np.asfarray(skid_data[['Average Velocity (mph)','Velocity Stdev','Average Acceleration (mph per s)', 'Max Acceleration (mph per s)', ' Acceleration Stdev','Displacement','Total Distance Traveled','Max Direction Change per sec', ' Direction Stdev','Time (s)', 'Turns', 'Aggressive Turns', 'Stops', 'Large Deceleration Events', 'Deceleration Events', 'Max Deceleration Event']])

#Correct missing data 
imputer = Imputer(missing_values="NaN", strategy="mean")
X = imputer.fit_transform(X)

###############################################################################
# Compute clustering
print("Compute unstructured hierarchical clustering...")
st = time.time()
ward = AgglomerativeClustering(n_clusters=6, linkage='ward').fit(X)
elapsed_time = time.time() - st
label = ward.labels_
print("Elapsed time: %.2fs" % elapsed_time)
print("Number of points: %i" % label.size)

###############################################################################
# Plot result
fig = plt.figure()
ax = p3.Axes3D(fig)
ax.view_init(7, -80)
for l in np.unique(label):
    ax.plot3D(X[label == l, 0], X[label == l, 1], X[label == l, 2],
              'o', color=plt.cm.jet(np.float(l) / np.max(label + 1)))
plt.title('Without connectivity constraints (time %.2fs)' % elapsed_time)


###############################################################################
# Define the structure A of the data. Here a 10 nearest neighbors
from sklearn.neighbors import kneighbors_graph
connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)

###############################################################################



