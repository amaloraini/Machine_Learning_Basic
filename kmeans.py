import pandas as pd 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import sys

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
	
def iris_clustering():
	df = pd.read_csv('Datasets/iris.csv')

	#create data and labels 
	data = df[['sepal_length', 'sepal_width', 'petal_length','petal_width']].values
	#data = df[['sepal_length', 'petal_length']].values
	labels = df['species'].values

	#Create and fit the model
	model = KMeans(n_clusters=3)
	model.fit(data)
	
	preds = model.predict(data)

	#scatter plot of the KMeans prediction 
	xs = data[:, 0] #sepal_length
	ys = data[:, 2] #petal length	


	
	plt.scatter(xs, ys, c=preds, alpha=0.5)
	#centers = model.cluster_centers_
	#centroids_x, centroids_y = centers[:, 0], centers[:, 1]
	#plt.scatter(centroids_x, centroids_y, marker='D', s=100, color='black', alpha=0.5)	
	plt.show()

	#Building crosstab between the predicted classes and the labels 
	ct = pd.DataFrame({'predictions':preds, 'species':labels})
	crosstab = pd.crosstab(ct['predictions'], ct['species'])
	print(crosstab)

	#Inertia measure is to gauge how good is Kmeans clustering (should be minimized)
	print(model.inertia_)


	#comparing different clusters and their interias 
	cluster_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
	interias = []
	for x in cluster_numbers:
		model = KMeans(n_clusters=x)
		model.fit(data)
		interias.append(model.inertia_)
	plt.plot(cluster_numbers, interias)
	plt.show()

def wine_clustering():
	#laod wine dataset	
	df = pd.read_csv('Datasets/wine.csv')
	print(df.columns)

	#notice that proline has large values compared with the others 
	print(df.info())	
	print(df.describe())

	#create data and labels 
	data = df[['alcohol', 'malic_acid', 'ash','alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids','nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue','od280', 'proline']].values
	labels = df[['class_label', 'class_name']].values
	model = KMeans(n_clusters=3)
	scaler = StandardScaler()
	pipeline = make_pipeline(scaler, model)

	pipeline.fit(data)
	preds = pipeline.predict(data)


	#crosstabe preds with labels 
	ct = pd.DataFrame({'Predictions': preds, 'Labels': labels[:, 1]})
	crosstab = pd.crosstab(ct['Predictions'], ct['Labels'])
	print(crosstab)

	#print intertia 
	print(model.inertia_)

def fish_clustering():
	#load and clean fish dataset
	df = pd.read_csv('Datasets/fish.csv')
	df.columns = ['Bream', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6']
	df['integer_Bream'] = df['Bream'].apply(recod_fish)
	print(df.columns)
	print(df.info())
	print(df.head())	
	
	#analyzing and studying the important attributes 
	df['p5'].plot.hist(label='23.0', color='blue')
	sns.pairplot(df, hue='Bream')
	plt.show()
	data = df[['p1', 'p2', 'p3', 'p4', 'p5', 'p6']].values	
	labels = df['Bream'].values
	colors = df['integer_Bream'].values
	plt.scatter(data[:, 3], data[:, 5], c=colors)
	plt.show()
	
	#create scaler and kmeans 
	scaler =StandardScaler()
	kmeans = KMeans(n_clusters=4)
	pipeline = make_pipeline(scaler, kmeans)

	#fit and train the data
	pipeline.fit(data)
	preds = pipeline.predict(data)

	#creating crosstab to evaluate and see inertia
	ct = pd.DataFrame({'Predictions':preds, 'Bream': labels})
	cross = pd.crosstab(ct['Predictions'], ct['Bream'])
	print(cross)
	print(kmeans.inertia_)


#iris_clustering()
#wine_clustering()
fish_clustering()