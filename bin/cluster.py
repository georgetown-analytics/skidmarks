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
from sklearn import linear_model
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn import preprocessing

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
    crime_data = pd.DataFrame.from_csv(in_data, sep=',')

#crime_data=crime_data.fillna(value=-999)

#load all numeric data into an array. The offense column from the crime data
#is excluded
as_array = np.asfarray(crime_data[['Velocity Stdev','Aggressive Turns']])#'Max Velocity', 'Velocity Stdev','Average Acceleration (mph per s)', 'Max Acceleration (mph per s)', ' Acceleration Stdev','Displacement','Total Distance Traveled','Max Direction Change per sec', ' Direction Stdev','Time (s)', 'Turns', 'Aggressive Turns', 'Stops', 'Large Deceleration Events', 'Deceleration Events', 'Max Deceleration Event']])

#number of groups
n_clusters=4

#Correct missing data 
imputer = Imputer(missing_values="NaN", strategy="mean")
patched = imputer.fit_transform(as_array)

# Preprocessing tricks
#patched = StandardScaler().fit_transform(patched)
#patched = scale(patched, axis=0, with_mean=True)

patched = preprocessing.normalize(patched, norm='l2')

#min_max_scaler = preprocessing.MinMaxScaler()
#patched = min_max_scaler.fit_transform(patched)



#Correct missing data 
#imputer = Imputer(missing_values=-999, strategy="mean")
#patched = imputer.fit_transform(as_array)

#cluster data 
cluster = KMeans(n_clusters=n_clusters)
cluster.fit(patched)

#assigned grouped labels to the crime data
labels = cluster.labels_


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
# Make Predictions
predictions = cluster.predict(patched)

SilouetteCoefficient = metrics.silhouette_score(patched, labels, metric='euclidean')

print "The Silouette Coefficient is", SilouetteCoefficient

model = sm.OLS(labels, patched)

results = model.fit()

print results.summary()

# Find centers
centers = cluster.cluster_centers_
center_colors = colors[:len(centers)]
plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)

#plt.subplot(1,4,idx+1)
plt.scatter(patched[:, 0], patched[:, 1], color=colors[predictions].tolist(), s=10)

plt.xticks(())
plt.yticks(())
plt.ylabel('$x_1$')
plt.xlabel('$x_0$')

plt.show()
