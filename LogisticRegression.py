# "Logistic Regression from Scratch"
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

#to make the experiment  reproducible, random.seed should be set up
np.random.seed(44)

def create_data():
	'''
	######################
	Creating dataset coordiantes randomly and set half the data to class 0
	and the other half to class 1. For demonstration purpose, I plotted the 
	data too
	'''
	number_of_data = 2000	

	cov = [[1, 0.7],[0.7, 1]] #default cov = [[1, 0.7],[0.7, 1]]

	#multivariate_normal is a generalized version of one dimension normal distribution (Gaussian dist.)
	#multivariate_normal(mean, diagonal covariance, total number)
	#mean represents the coordinate, cov indicates how to variables vary
	x1 = np.random.multivariate_normal([0, 0], cov, number_of_data)
	x2 = np.random.multivariate_normal([1, 4], cov, number_of_data)

	#vstack add two arrays vertically together, hstack add two data together horizontally 
	data = np.vstack((x1, x2)).astype(np.float32)

	y1 = np.zeros(number_of_data)
	y2 = np.ones(number_of_data)
	labels = np.hstack((y1, y2))

	#set up the figure and show it
	#plt.figure(figsize=(12,8))
	#plt.scatter(data[:, 0], data[:, 1], c = labels, alpha=0.5)
	#plt.show()
	return data, labels

#sigmoid function
def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))

#loss function
def log_likelihood(features, target, weights):
    scores = np.dot(features, weights)
    ll = np.sum( target*scores - np.log(1 + np.exp(scores)) )
    return ll

def logistic_regression(features, target, num_steps, learning_rate, add_intercept = False):
    
    #this is B in our function
    if add_intercept:
        intercept = np.ones((features.shape[0], 1))
        features = np.hstack((intercept, features))
    
    #initialize weights ranomly
    weights = np.random.random((features.shape[1]))

    print('weights')
    print(weights)

    for step in range(num_steps):
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)

        # Update weights with gradient
        output_error_signal = target - predictions
        gradient = np.dot(features.T, output_error_signal)
        weights += learning_rate * gradient
        
        # Print log-likelihood every so often
        if step % 1000 == 0:
            _ = log_likelihood(features, target, weights)
            print (_)
        
    return weights

#main class
data, labels = create_data()
weights = logistic_regression(data, labels, num_steps = 10000, learning_rate = 0.001, add_intercept=True)

#test our original data with the gained weights
test_data = np.hstack((np.ones((data.shape[0], 1)), data))
final_scores = np.dot(test_data, weights)
preds1 = np.round(sigmoid(final_scores))


#sklearn Logistic Regression
clf = LogisticRegression(fit_intercept=True, C = 1e15)
clf.fit(data, labels)
preds2 = clf.predict(data)

#draw the decision boundary line 
coef = clf.coef_
print(coef)
intercept = clf.intercept_
print(intercept)
ex1 = np.linspace(-3, 4)
ex2 = -(coef[:,0] * ex1 + intercept[0]) / coef[:,1]
plt.figure(figsize=(12,8))
plt.scatter(data[:, 0], data[:, 1], c = labels, alpha=0.5)
plt.plot(ex1, ex2, color='r', label='decision boundary');
plt.show()

#classification report
target_names = ['Class 0', 'Class 1']
print("===The implemented Logistic Regression===")
print(classification_report(labels, preds1, target_names=target_names))

#compared with Sklearn Logisition regression
print("===Sklearn Logistic Regression===")
print(classification_report(labels, preds2, target_names=target_names))