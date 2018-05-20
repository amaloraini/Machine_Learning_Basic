from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from sklearn.preprocessing import normalize, scale
import numpy as np
import sys
#original from https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/

np.random.seed(444)  # for repeatability of this tutorial
n = 20
cov = [[3, 1], [1, 4]]
a = np.random.multivariate_normal([10, 0], cov, size=n)
b = np.random.multivariate_normal([0, 20], cov, size=n)
X = np.concatenate((a, b),)
#X = normalize(X) #X=scale(X)
print (X.shape)  # 150 samples with 2 dimensions
plt.scatter(X[:,0], X[:,1])
plt.show()

# generate the linkage matrix
merg = linkage(X, method='complete') #other parameters ward, single, average and also distance metrics (cos similarity, euclidean distance..)

c, coph_dists = cophenet(merg, pdist(X))

#[idx1, idx2, dist, sample_count].
print(merg)

idxs = [0, 13, 4, 8]
plt.figure(figsize=(10, 8))
plt.scatter(X[:,0], X[:,1])  # plot all points
plt.scatter(X[idxs,0], X[idxs,1], c='r')  # plot interesting points in red again
plt.show()

# calculate full dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(merg,leaf_rotation=90.,  # rotates the x axis labels
leaf_font_size=8.,)  # font size for the x axis labels
plt.show()
