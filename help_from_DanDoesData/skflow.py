import tensorflow.contrib.learn as skflow
#sklearn has premade datasets like mnist
import sklearn
from sklearn import datasets, metrics, preprocessing
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.ion()

#easier way to access mnist
mnist = datasets.load_digits()
#seet actually gets to look at it
plt.pcolor(mnist.data[55].reshape(8,8), cmap=plt.cm.gray_r)
#mnist.target
#len(_)

#github's linear classifier- replaced with non deprecated linearclassifier
iris = datasets.load_iris()
classifier = skflow.LinearClassifier(n_classes=3)
classifier.fit(iris.data, iris.target)
score = metrics.accuracy_score(iris.target, classifier.predict(iris.data))
print("Accuracy: %f" % score)

#github's linear regressor model- replaced with non deprecated linearregrssor
boston = datasets.load_boston()
X = preprocessing.StandardScaler().fit_transform(boston.data)
regressor = skflow.LinearRegressor()
regressor.fit(X, boston.target)
score = metrics.mean_squared_error(regressor.predict(X), boston.target)
print ("MSE: %f" % score)

#dan and dave's classifier
classifier = skflow.TensorFlowDNNClassifier(hidden_units=[5,10,15,20], n_classes = len(mnist['target_names']), learning_rate=.05, steps=10000)
classifier.fit(mnist.data, mnist.target)
C = classifier.fit(mnist.data, mnist.target)
score = metrics.accuracy_score(mnist.target, classifier.predict(mnist.data))
print("Accuracy: %f" % score)

#seet actually looks at some weights
num_layers = len(classifier.hidden_units)
f, con = plt.subplots(num_layers)
for layer in range(len(num_layers):
	con[layer].pcolor(classifier.weights_[layer], cmap=plt.cm.gray_r)

	con[xx,yy].pcolor(ww2[yy*8+xx][0], cmap=plt.cm.gray_r)










