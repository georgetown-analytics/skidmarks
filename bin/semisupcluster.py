# -*- coding: utf-8 -*-

###############################################################################
# Information
###############################################################################
# Created by Linwood Creekmore 

# From Mastering Machine Learning with scikit-learn, page 132-133

# In partial fulfillment of the requirements for the Georgetown University Data Analytics Graduate Certificate Program

# May 8, 2015

# https://plus.google.com/+LinwoodCreekmoreIII/

###############################################################################
# Imports
###############################################################################


import os
import numpy as np
import pandas as pd 
from sklearn.metrics import * 
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import MiniBatchKMeans

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
as_array = np.asfarray(skid_data[:int(train)][['Average Velocity (mph)','Turns']]) #'Velocity Stdev','Average Acceleration (mph per s)', 'Max Acceleration (mph per s)', ' Acceleration Stdev','Displacement','Total Distance Traveled','Max Direction Change per sec', ' Direction Stdev','Time (s)', 'Turns', 'Aggressive Turns', 'Stops', 'Large Deceleration Events', 'Deceleration Events', 'Max Deceleration Event']])
X_test = np.asfarray(skid_data[int(test):][['Average Velocity (mph)','Turns']]) #'Velocity Stdev','Average Acceleration (mph per s)', 'Max Acceleration (mph per s)', ' Acceleration Stdev','Displacement','Total Distance Traveled','Max Direction Change per sec', ' Direction Stdev','Time (s)', 'Turns', 'Aggressive Turns', 'Stops', 'Large Deceleration Events', 'Deceleration Events', 'Max Deceleration Event']])

###############################################################################
# scikit-learn preprocessing; uncomment to see differences
###############################################################################

#Correct missing data 
imputer = Imputer(missing_values="NaN", strategy="mean")
patched = imputer.fit_transform(as_array)
patched2 = imputer.fit_transform(X_test)

# Preprocessing tricks
#patched = StandardScaler().fit_transform(patched)
#patched = scale(patched, axis=0, with_mean=True)
#patched = preprocessing.normalize(patched, norm='l2')
#min_max_scaler = preprocessing.MinMaxScaler()
#patched = min_max_scaler.fit_transform(patched)

###############################################################################
# Constants and Set Values
###############################################################################

X_train = patched
X_test = patched2

###############################################################################
# Main Functions
###############################################################################

estimator = MiniBatchKMeans(n_clusters = 5)
estimator.fit_transform(X_train)
y_train = estimator.labels_

clf = LogisticRegression(C=0.001, penalty ='l2')
clf.fit_transform(X_train,y_train)
predictions = clf.predict(X_test)

print classification_report(y_train,predictions)
print 'Precision: ', precision_score(y_train,predictions)
print 'Recall: ', recall_score(y_train, predicitons)
print 'Accuracy: ', accuracy_score(y_train,predictions)