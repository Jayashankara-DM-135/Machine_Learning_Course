#Data Preproccssing

#Importing libarary 
import numpy as np # for math related stuffs
import matplotlib.pyplot as plt # Ploting graphs ext
import pandas as pd #handling data sets

#import the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

#Splitting the dataset into the Traning set and Test set.
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)


#Fitting simple Linear Regression to the traning set.
"""
Here we are creating machine called "regressor" based simple regression learning.
"""
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)


#Predicting the Test set results.
"""
Here we pass Test data to machine "reggressopr"
"""
y_pred = regressor.predict(X_test)

#Visualising the Traning set results
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title("Salary V/S Experience (Traning set)")
plt.xlabel("Years of expirence")
plt.ylabel("Salary")
plt.show()

plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title("Salary V/S Experience (Test set)")
plt.xlabel("Years of expirence")
plt.ylabel("Salary")
plt.show()





