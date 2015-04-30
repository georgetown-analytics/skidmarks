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
from sklearn.cluster import MiniBatchKMeans
from collections import OrderedDict
from bokeh.plotting import output_file
from sklearn.preprocessing import Imputer

print(__doc__)

from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

df = pd.read_csv('./lin.csv')

digits = np.asfarray(df[['Average Velocity (mph)','Average Acceleration (mph per s)', 'Max Velocity', 'Velocity Stdev', 'Average Acceleration (mph per s)', 'Max Acceleration (mph per s)', ' Acceleration Stdev', 'Max Direction Change per sec', ' Direction Stdev', 'Time (s)']])
data = scale(digits.data)

n_samples, n_features = data.shape, digits.shape
n_digits = len(digits)


sample_size = len(df)

#print("n_digits: %d, \t n_samples %d, \t n_features %d"
      #% (n_digits, n_samples, n_features))


print(79 * '_')
#print('% 9s' % 'init'
     # '    time  inertia    homo   compl  v-meas     ARI AMI  silhouette')


def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('% 9s   %.2fs    %i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))

bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
              name="k-means++", data=data)

bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),
              name="random", data=data)

# in this case the seeding of the centers is deterministic, hence we run the
# kmeans algorithm only once with n_init=1
pca = PCA(n_components=n_digits).fit(data)
bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
              name="PCA-based",
              data=data)
print(79 * '_')

###############################################################################
# Visualize the results on PCA-reduced data

reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
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
    n_clusters=5
    
    #Correct missing data 
    imputer = Imputer(missing_values=-999, strategy="mean")
    patched = imputer.fit_transform(as_array)
   
    #cluster data 
    cluster = MiniBatchKMeans(n_clusters=n_clusters)
    #cluster = KMeans(n_clusters=n_clusters)
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

    reduced_data = decomposition.PCA(n_components=2).fit_trans4form(as_array)
    kmeans = KMeans(init='k-means++', n_clusters=4, n_init=10)
    kmeans.fit(reduced_data)
    
    from sklearn import decomposition
    pca = decomposition.PCA(n_components=9)
    pca.fit(as_array)
    PCA(copy=True, n_components=9, whiten=True)
    X = pca.transform(as_array)
    
    #Plotting crap
    centers = cluster.cluster_centers_
    center_colors = colors[:len(centers)]
    plt.scatter(centers[:, 0], centers[:, 1], s=50, c=center_colors)
    
    #plt.subplot(1,4,idx+1)
    plt.scatter(X[:, 0], X[:, 1], color=colors[predictions].tolist(), s=10)

    #plt.scatter( )
    plt.show()
if __name__ == "__main__":
    main()

'''