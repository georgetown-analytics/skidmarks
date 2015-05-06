from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import scale
from sklearn.preprocessing import Imputer
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold


with open("./lin.csv") as in_data:
    skid_data = pd.DataFrame.from_csv(in_data, sep=',')

#crime_data=crime_data.fillna(value=-999)

#load all numeric data into an array. The offense column from the crime data
#is excluded
X = np.asfarray(skid_data[['Average Velocity (mph)','Turns','Max Velocity', 'Velocity Stdev','Average Acceleration (mph per s)', 'Max Acceleration (mph per s)', ' Acceleration Stdev','Displacement','Total Distance Traveled','Max Direction Change per sec', ' Direction Stdev','Time (s)', 'Turns', 'Aggressive Turns', 'Stops', 'Large Deceleration Events', 'Deceleration Events', 'Max Deceleration Event']])


#Correct missing data 
imputer = Imputer(missing_values="NaN", strategy="mean")
X = imputer.fit_transform(X)

print "Before feature selection",X.shape

# Preprocessing tricks
#X= StandardScaler().fit_transform(X)
#patched = scale(patched, axis=0, with_mean=True)

#patched_normalized = preprocessing.normalize(patched, norm='l2')

#min_max_scaler = preprocessing.MinMaxScaler()
#patched_minmax = min_max_scaler.fit_transform(patched)

selector = VarianceThreshold()
selector.fit_transform(X)

print "After feature selection",X.shape


'''
X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
X_new.shape


print "Trying the kfolds"
>>> from sklearn import cross_validation
>>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
>>> y = np.array([1, 2, 3, 4])
>>> kf = cross_validation.KFold(4, n_folds=2)
>>> len(kf)
2
>>> print(kf)  
sklearn.cross_validation.KFold(n=4, n_folds=2, shuffle=False,
                               random_state=None)
>>> for train_index, test_index in kf:
...    print("TRAIN:", train_index, "TEST:", test_index)
...    X_train, X_test = X[train_index], X[test_index]
...    y_train, y_test = y[train_index], y[test_index]
TRAIN: [2 3] TEST: [0 1]
TRAIN: [0 1] TEST: [2 3]
.. automethod:: __init__
'''