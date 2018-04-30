#Decision Trees
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(44)

#read the dataset
df = pd.read_csv('Datasets/iris.csv')

#Check the status of the dataset and if there is any missing data
print(df.info()) # df.isnyll().any()

#check interesting values such as petal_width and petal_length
print(df.describe())

#show the frequency of the data
df['petal_width'].plot.hist(label='petal_width', color='blue')
df['sepal_length'].plot.hist(label='sepal_length', color='red')
plt.show()

#This function will create a grid of axes such that each variable in the data is shared
#in the y-axis and in the x-axis 
sns.set(style="ticks")
sns.pairplot(df, hue="species")
plt.show()

#read and split the data into 0.8 and 0.2
data = df[['sepal_length', 'sepal_width', 'petal_length','petal_width']].values
classes = df['species'].values

test_ratio = 0.2
x_train, x_test, y_train, y_test = train_test_split(data, classes, test_size=test_ratio, shuffle= True, random_state=44)

#Decision Tree classifier 
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
preds1 = dtc.predict(x_test)

#print classification report
print('===Using a single decision tree===')
print(classification_report(y_test, preds1))#, target_names=target_names))

#Random forest with 5 decision trees
dtr = RandomForestClassifier()
dtr = RandomForestClassifier(n_estimators = 5)
dtr.fit(x_train, y_train)
preds2 = dtr.predict(x_test)

#print classification report
print('===Using random forest with 5 decision trees===')
print(classification_report(y_test, preds2))

