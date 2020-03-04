# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 21:22:24 2020

@author: jmallesh
"""
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense

from matplotlib import pyplot
from numpy import mean
from numpy import std
import numpy
from numpy import array
from numpy import argmax


#Evaluate single model
def evaluate_model(trainX, trainY, testX, testY):
    #one hot encode for class
    trainy_enc = to_categorical(trainY)
    testy_enc = to_categorical(testY)
    
    #define model.
    model = Sequential()
    model.add(Dense(50, input_dim=2, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # fit a  model
    model.fit(trainX, trainy_enc, epochs=50, verbose=0)
    
    #Evaluate the model.
    _, test_acc = model.evaluate(testX, testy_enc, verbose=0)
    return model, test_acc


# make an Ensemble predication for multi-class classification.
def ensemble_predication(members, testX):
    #make predications.
    yhats = [model.predict(testX) for model in members]
    yhats = array(yhats)
    print("Len :{} => {}".format(len(yhats), yhats))
    summed = numpy.sum(yhats, axis=0)
    print("Summed=> {}".format(summed))
    results = argmax(summed, axis=1)
    print("Result=> {}".format(results))
    return results

#Evaluate a specific number of members in an ensemble
def evaluate_n_members(members, n_members, testX, testY):
    subset = members[:n_members]
    yhat = ensemble_predication(subset, testX)
    print("====>len:{} {}".format(len(yhat), yhat))
    return accuracy_score(testY, yhat)

#Generate 2d classification dataset.
dataX, dataY = make_blobs(n_samples=55000, centers=3, n_features=2, cluster_std=2, random_state=2)
X, newX = dataX[:5000, :], dataX[5000:, :]
Y, newY = dataY[:5000], dataY[5000:]


#multiple train_test splits
n_splits = 10
scores, members = list(), list()

for _ in range(n_splits):
    trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.10)
    #evaluate a model
    model, test_acc = evaluate_model(trainX, trainY, testX, testY)
    print('Model : >>> %.3f' % test_acc)
    scores.append(test_acc)
    members.append(model)

#Summeraize the expected performance
print("Estimated Accuracy %.3f (%.3f)" % (mean(scores), std(scores)))

#Evaluate different number of ensemble on hold out set.
single_scores, ensemble_scores = list(), list()
for i in range(1, n_splits+1):
    ensemble_score = evaluate_n_members(members, i, newX, newY)
    newy_enc = to_categorical(newY)
    _, single_score = members[i-1].evaluate(newX, newy_enc, verbose=0)
    print('>>> %d: Single_score: %.3f, ensemble_score: %.3f' % (i, single_score, ensemble_score))
    ensemble_scores.append(ensemble_score)
    single_scores.append(single_score)

#plot score vs number of ensemble menvers:
print("Accuracy %.3f (%.3f)" %(mean(single_scores), std(single_scores)))

x_axis = [i for i in range(1, n_splits+1)]
pyplot.plot(x_axis, single_scores, marker='o', linestyle='None')
pyplot.plot(x_axis,  ensemble_scores, marker='o')
pyplot.show()    
      
    

    
    
    

    
    
    

    
    






