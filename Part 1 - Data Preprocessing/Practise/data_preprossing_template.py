#Data Preproccssing
#Importing libarary 

import numpy as np # for math related stuffs
import matplotlib.pyplot as plt # Ploting graphs ext
import pandas as pd #handling data sets

#import the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

#Splitting the dataset into the Traning set and Test set.
from sklearn.model_selection import train_test_split
X_train, X_test, Y_tarin, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)





