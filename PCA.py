from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd

def recod_fish(label):
	if label == 'Roach':
		return 0
	elif label == 'Smelt':
		return 1
	elif label == 'Pike':
		return 2
	elif label == 'Bream':
		return 3
	else:
		return "ERROR"


df = pd.read_csv(('Datasets/fish.csv'), names = ['type','p1','p2', 'p3', 'p4', 'p5', 'p6'])

data = df[['p1','p2', 'p3', 'p4', 'p5', 'p6']].values
labels = df[['type']].values
colors = df['type'].apply(recod_fish)

#PCA to find the variance between attributes 
scaler = StandardScaler()
pca = PCA()
pipeline = make_pipeline(scaler, pca)
pipeline.fit(data)
print(pca.n_components_)
pipeline = make_pipeline(scaler, pca)
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.show()

#to find the correlation between two features 
correlation, pvalue = pearsonr(data[:,4], data[:, 3])
print(correlation)

#Dimension reduction with PCA
pca = PCA(n_components=2)
pca.fit(data)
pca_features = pca.transform(data)
xs = pca_features[:, 0]
ys = pca_features[:, 1]
plt.scatter(xs, ys, c = colors)
#plt.scatter(data[:,4], data[:, 3], c = colors)
plt.show()

#Try tsne 
tsne = TSNE(learning_rate=100)
transformed = tsne.fit_transform(data)
xs = transformed[:, 0]
ys = transformed[:, 1]
plt.scatter(xs, ys, c = colors)
plt.show()
