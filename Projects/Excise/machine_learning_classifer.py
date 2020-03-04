# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 20:28:52 2020

@author: jmallesh
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris

import argparse

#Build argument parser and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="Knn", 
                help="Enter Machine learning model to use")
args = vars(ap.parse_args())

models={
        'Knn':KNeighborsClassifier(n_neighbors=1),
        'naive_bayes':GaussianNB(),
        'logit':LogisticRegression(solver='lbfgs', multi_class='auto'),
        'svm':SVC(kernel='rbf', gamma='auto'),
        'tree':DecisionTreeClassifier(),
        'random_forest':RandomForestClassifier(n_estimators=100),
        'nn':MLPClassifier()
        }

#Load the dataset and split dataset into test and training dataset.
print("Loading dataset!!!")
dataset = load_iris()
(testX, trainX, testY, trainY) = train_test_split(dataset.data, dataset.target,
random_state=3, train_size=0.25)

#Build model
print("Preparing Model: {}".format(models[args['model']]))
model = models[args['model']]
model.fit(trainX, trainY)

#Evaluate the model
predication = model.predict(testX)
print(classification_report(testY, predication, 
                            target_names=dataset.target_names))



