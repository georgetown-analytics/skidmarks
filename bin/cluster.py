print(__doc__)

from time import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.preprocessing import Imputer
<<<<<<< HEAD
<<<<<<< HEAD
=======
from sklearn import linear_model
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
>>>>>>> master
=======

from sklearn import preprocessing

from sklearn import linear_model
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
>>>>>>> d17a7f5a32f40703be4c05ceb48dc00371e8ef90
from sklearn import preprocessing


# Some colors for later
colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)

###
 #load data from a CSV to a dataframe
with open("./lin.csv") as in_data:
    skid_data = pd.DataFrame.from_csv(in_data, sep=',')

#skid_data=skid_data.fillna(value=-999)

#load all numeric data into an array. The offense column from the crime data
#is excluded
<<<<<<< HEAD
<<<<<<< HEAD
=======

>>>>>>> d17a7f5a32f40703be4c05ceb48dc00371e8ef90
as_array = np.asfarray(skid_data[['Average Velocity (mph)','Turns','Max Velocity', 'Velocity Stdev','Average Acceleration (mph per s)', 'Max Acceleration (mph per s)', ' Acceleration Stdev','Displacement','Total Distance Traveled','Max Direction Change per sec', ' Direction Stdev','Time (s)', 'Turns', 'Aggressive Turns', 'Stops', 'Large Deceleration Events', 'Deceleration Events', 'Max Deceleration Event']])

#as_array = np.asfarray(crime_data[['Average Velocity (mph)','Displacement']])#'Max Velocity', 'Velocity Stdev','Average Acceleration (mph per s)', 'Max Acceleration (mph per s)', ' Acceleration Stdev','Displacement','Total Distance Traveled','Max Direction Change per sec', ' Direction Stdev','Time (s)', 'Turns', 'Aggressive Turns', 'Stops', 'Large Deceleration Events', 'Deceleration Events', 'Max Deceleration Event']])


#number of groups
n_clusters=4

# Preprocessing tricks
=======
as_array = np.asfarray(crime_data[['Average Velocity (mph)','Aggressive Turns']])#'Max Velocity', 'Velocity Stdev','Average Acceleration (mph per s)', 'Max Acceleration (mph per s)', ' Acceleration Stdev','Displacement','Total Distance Traveled','Max Direction Change per sec', ' Direction Stdev','Time (s)', 'Turns', 'Aggressive Turns', 'Stops', 'Large Deceleration Events', 'Deceleration Events', 'Max Deceleration Event']])

#number of groups
n_clusters=4
>>>>>>> master

#Correct missing data 
imputer = Imputer(missing_values="NaN", strategy="mean")
patched = imputer.fit_transform(as_array)
#min_max_scaler = preprocessing.MinMaxScaler()
#patched = min_max_scaler.fit_transform(patched)
#patched = preprocessing.normalize(patched, norm='l2') 

<<<<<<< HEAD
#patched = preprocessing.Binarizer().fit(patched)
#patched = binarizer.transform(as_array)
#patched = scaler.transform(as_array)
=======
# Preprocessing tricks
#patched = StandardScaler().fit_transform(patched)
#patched = scale(patched, axis=0, with_mean=True)

patched = preprocessing.normalize(patched, norm='l2')

#min_max_scaler = preprocessing.MinMaxScaler()
#patched = min_max_scaler.fit_transform(patched)
<<<<<<< HEAD
=======


>>>>>>> d17a7f5a32f40703be4c05ceb48dc00371e8ef90
>>>>>>> master


#cluster data 
cluster = KMeans(n_clusters=n_clusters)
cluster.fit_transform(patched)



#assigned grouped labels to the crime data
labels = cluster.labels_
<<<<<<< HEAD
<<<<<<< HEAD
=======

>>>>>>> d17a7f5a32f40703be4c05ceb48dc00371e8ef90
skid_data["labels"]=labels




<<<<<<< HEAD
'''
  # Fit the model with our algorithm
model = MiniBatchKMeans(n_clusters=3)
model.fit(as)
'''
=======
#copy dataframe (may be memory intensive but just for illustration)
skid_data = crime_data.copy()
#print pd.Series(classified_data)
#print pd.Series(prediction_data)
skid_data['Cluster Class'] = pd.Series(labels, index=skid_data.index)
print skid_data.describe()
print skid_data
#print list(skid_data.columns)
skid_data.plot( x = 'Aggressive Turns', y = 'Cluster Class', kind = 'scatter')
plt.show()
>>>>>>> master
# Make Predictions
predictions = cluster.predict(patched)

SilouetteCoefficient = metrics.silhouette_score(patched, labels, metric='euclidean')

<<<<<<< HEAD
<<<<<<< HEAD
=======

>>>>>>> d17a7f5a32f40703be4c05ceb48dc00371e8ef90
print "The Silouette Coefficient score is", SilouetteCoefficient

print "The Silouette Coefficient is", SilouetteCoefficient

model = sm.OLS(labels, patched)

results = model.fit()

print results.summary()
<<<<<<< HEAD
>>>>>>> master
=======
master
>>>>>>> d17a7f5a32f40703be4c05ceb48dc00371e8ef90

# Find centers
centers = cluster.cluster_centers_
center_colors = colors[:len(centers)]
plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)

#plt.subplot(1,4,idx+1)
plt.scatter(patched[:, 0], patched[:, 1], color=colors[predictions].tolist(), s=10)

plt.xticks(())
plt.yticks(())
plt.ylabel('$Feature A$')
plt.xlabel('$Feature B$')

plt.show()
