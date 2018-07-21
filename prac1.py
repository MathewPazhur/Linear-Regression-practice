# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 19:00:05 2018

@author: Mathew
"""

# Importing the libraries
import numpy as np
from scipy import sparse
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from matplotlib import style

# Importing the dataset
dataset = pd.read_csv('dataset-prac1.csv')
X = dataset.iloc[:, -1:].values
y = dataset.iloc[:, 0].values

print(X)
print(y)

#displaying list of keys 

print("Keys of dataset: \n{}".format(dataset.keys()))


## Splitting the data into test and train data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

#Displaying shape of above mentioned arrays

print("X_train shape : {}".format(X_train.shape))
print("X_test shape : {}".format(X_test.shape))
print("y_train shape : {}".format(y_train.shape))
print("y_test shape : {}".format(y_test.shape))


#Fitting simple linear regression he the training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# Predicitng the test set results

y_pred = regressor.predict(X_test)


# Visualizing the training set results

plt.scatter(X_train,y_train,color='red')
plt.plot(X_train, regressor.predict(X_train), color= 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of experience')
plt.xlabel('Salary')
plt.show()