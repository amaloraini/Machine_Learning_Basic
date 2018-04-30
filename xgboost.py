import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.datasets import dump_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import datasets
from sklearn.metrics import precision_score
from sklearn.externals import joblib

np.random.seed(44)

#read the dataset
df = pd.read_csv('Datasets/iris.csv')

#map labels to integers 
classes = [x for x in df['species']]
classes = set(classes)
print(classes)
dic = dict(zip(classes, range(0,3)))
df['species'] = df['species'].map(dic, na_action='ignore')

#assign data and classes 
data = df[['sepal_length', 'sepal_width', 'petal_length','petal_width']].values
classes = df['species'].values

#split data 
test_ratio = 0.2
x_train, x_test, y_train, y_test = train_test_split(data, classes, test_size=test_ratio, shuffle= True, random_state=44)

#putting the training and test datasets into matrices 
dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)

'''
#more efficient memory consumption#

dump_svmlight_file(X_train, y_train, 'dtrain.svm', zero_based=True)
dump_svmlight_file(X_test, y_test, 'dtest.svm', zero_based=True)
dtrain_svm = xgb.DMatrix('dtrain.svm')
dtest_svm = xgb.DMatrix('dtest.svm')
'''
# set xgboost params
param = {
    'max_depth': 3,  # the maximum depth of each tree
    'eta': 0.3,      # the training step for each iteration
    'silent': 1,     # logging mode - quiet
    'objective': 'multi:softprob',  # error evaluation for multiclass training
    'num_class': 3}  # the number of classes that exist in this datset
iter = 100  # the number of training iterations

bst = xgb.train(param, dtrain, iter)

#to see how the model looks like
bst.dump_model('temp/dump.raw.txt')

#predicting the test data, output [x1 x2 x3]
model_preds = bst.predict(dtest)
preds = np.asarray([np.argmax(line) for line in model_preds])

'''
# to skip all the above settings, you can run the following default settings#

model = XGBClassifier()
model.fit(x_train, y_train)
preds = model.predict(x_test)
'''

# evaluate predictions
target_names = list(dic.keys())
accuracy = accuracy_score(y_test, preds)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print('===Using xgboost===')
print(classification_report(y_test, preds, target_names=target_names))

#To save the model 
#joblib.dump(bst, 'bst_model.pkl', compress=True)

