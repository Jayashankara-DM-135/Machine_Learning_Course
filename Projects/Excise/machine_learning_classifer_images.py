# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 22:34:43 2020

@author: jmallesh
"""
#Machine learning models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

#Convert texual labels into numbers.
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#Basic image processing Libaray pillow/PIL
from PIL import Image
import numpy as np
import os
import argparse
from imutils import paths


def extract_color_stats(image):
    """
    Split the image into RGB chanels and create a feature vector with
    6 values.
    """
    (R, G, B) = image.split()
    features = [np.mean(R), np.mean(G), np.mean(B), 
                np.std(R), np.std(G), np.std(B)]
    return features

#Add the argumnet parser and parse the args.
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', type=str, default='3scenes', 
                help="Provide the path of dataset")
ap.add_argument('-m', '--model', type=str, 
                default='knn', help="Enter machine learning model")
args = vars(ap.parse_args())

models = {
        'knn':KNeighborsClassifier(n_neighbors=1),
        'naive_bais':GaussianNB(),
        'logit':LogisticRegression(solver='lbfgs', multi_class='auto'),
        'svm':SVC(kernel='rbf', gamma='auto'),
         'tree':DecisionTreeClassifier(),
         'random_forest':RandomForestClassifier(n_estimators=100),
         'nn':MLPClassifier()
        }

#Extarct the image data from disk.
imagePaths = paths.list_images(args['dataset'])
print(os.getcwd())
print(args['dataset'])
print("Image path is:")
print(imagePaths)

for i in imagePaths:
    if i == 1:
        print(i)
    
data = []
labels = []

for imagePath in imagePaths:
    image = Image.open(imagePath)
    features = extract_color_stats(image)
    data.append(features)
    
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

#Encode the labels
le = LabelEncoder()
labels = le.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, 
 test_size=0.20)


model = models[args['model']]
model.fit(testX, testY)


predications = model.predict(testX)
print(classification_report(testY, predications, target_names=le.classes_))

    




