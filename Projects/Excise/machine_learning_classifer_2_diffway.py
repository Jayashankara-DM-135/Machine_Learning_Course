# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 14:02:51 2020

@author: jmallesh
"""

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC


import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names =['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)


#Dimensions of the dataset
print(dataset.shape)
#Peek at the data itself.
print(dataset.head(5))
#Statistical summary of all attributes.
print(dataset.describe())
#Breakdown of the data by the class variable
print(dataset.groupby('class').size())
#Box and whisper plot.
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

#histogram
dataset.hist()
plt.show()

#scatter_matrix
scatter_matrix(dataset)
plt.show()


#Spot-check the Algorithms.
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
#models.append(('CART/Tree', DecisionTreeClassifier))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
models.append(('NN', MLPClassifier()))
models.append(('Ensemble', RandomForestClassifier()))


#Split the data.
array = dataset.values
test = array[:,0:4]
train = array[:, 4]
(testX, trainX, testY, trainY) = train_test_split(test, train, 
test_size=0.20, random_state=1, shuffle=True)


results = []
names = []

for name, module in models:
    kflod = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    cv_results = cross_val_score(module, testX, testY, cv=kflod, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print("%s %f:%f" %(name, cv_results.mean(),  cv_results.std()))

#
plt.boxplot(results, labels=names)
plt.title("Algorithm Comparison'")
plt.show()
    

#From above expirement looks like SVM work fine.

model = SVC(gamma='auto')
model.fit(trainX, trainY)
predication = model.predict(testX)


print(accuracy_score(testY, predication))
print(confusion_matrix(testY, predication))
print(classification_report(testY, predication))






