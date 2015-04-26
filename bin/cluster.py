import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from conf import settings
from datetime import datetime
from sklearn.decomposition import PCA
#from bokeh.charts import Scatter
from sklearn.cluster import KMeans
from collections import OrderedDict
from bokeh.plotting import output_file
from sklearn.preprocessing import Imputer
'''
#Create a timestamp for the output file
def time_stamp():
    """
    Creates mm-dd-yyyy timestamp
    """
    now = time.time()
    return datetime.fromtimestamp(now).strftime("%m%d%Y")
    
def create_ordered_dict(data_frame, group_by, x_field="X", y_field="Y"):
    """
    puts data into a dicitonary for Bokeh. 
    """
    pdict = OrderedDict()
    
    g = data_frame.groupby(group_by)
    
    for i in g.groups.keys():
        lat = getattr(g.get_group(i), y_field)
        lng = getattr(g.get_group(i), x_field)
        pdict[i] = zip(lat,lng)
    return pdict
 '''   
def main():
    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)
    """
    This function will
        -load data from a csv
        -impute missing data with the column's mean
        -perform kmneans clustering
        -produce an html scatter plot
    """
    
    #load data from a CSV to a dataframe
    with open('./lin.csv') as in_data:
        crime_data = pd.DataFrame.from_csv(in_data, sep=',')
    
    crime_data=crime_data.fillna(value=-999)
    
    #load all numeric data into an array. The offense column from the crime data
    #is excluded
    as_array = np.asfarray(crime_data[['Average Velocity (mph)','Average Acceleration (mph per s)', 'Max Velocity', 'Velocity Stdev', 'Average Acceleration (mph per s)', 'Max Acceleration (mph per s)', ' Acceleration Stdev', 'Max Direction Change per sec', ' Direction Stdev', 'Time (s)']])
    
    #number of groups
    n_clusters=4
    '''
    #Correct missing data 
    imputer = Imputer(missing_values=-999, strategy="mean")
    patched = imputer.fit_transform(as_array)
    '''
    #cluster data 
    cluster = KMeans(n_clusters=n_clusters)
    cluster.fit(as_array)

    print cluster
    print as_array

    predictions = cluster.predict(as_array)
    
    #assigned grouped labels to the crime data
    labels = cluster.labels_
    crime_data["labels"]=labels

###############################################################################
# PCA Reduction
###############################################################################

    ''' reduced_data = decomposition.PCA(n_components=2).fit_trans4form(as_array)
    kmeans = KMeans(init='k-means++', n_clusters=4, n_init=10)
    kmeans.fit(reduced_data)'''
    
    from sklearn import decomposition
    pca = decomposition.PCA(n_components=2)
    pca.fit(as_array)
    PCA(copy=True, n_components=2, whiten=False)
    X = pca.transform(as_array)
    
    #Plotting crap
    centers = cluster.cluster_centers_
    center_colors = colors[:len(centers)]
    plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)
    
    #plt.subplot(1,4,idx+1)
    plt.scatter(as_array[:, 0], as_array[:, 6], color=colors[predictions].tolist(), s=10)

    #plt.scatter( )
    plt.show()
if __name__ == "__main__":
    main()

