#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import csv
import spotipy
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load the data and split into 3 equal sets
data = pd.read_csv('DatasetWithArtistPopularity.csv', header=0)
test, valid, train = np.split(data, [int(.15*len(data)), int(.3*len(data))])
train = np.array(train)
valid = np.array(valid)
train_x = np.array(train[:, 3:-1]).astype(np.float)
train_y = np.array(train[:, -1]).astype(np.float)
valid_x = np.array(valid[:, 3:-1]).astype(np.float)
valid_y = np.array(valid[:, -1]).astype(np.float)

# Apply linear regression model
model = LinearRegression()
model.fit(train_x, train_y)
y_pred = model.predict(valid_x)
### YOUR CODE HERE - Calculate the RMSE. Report and plot the curve on the validation set.
print('poly train mse: ', mean_squared_error(model.predict(train_x), train_y))
print('poly valid mse: ', mean_squared_error(y_pred, valid_y))



### YOUR CODE HERE - Fit a 10-degree polynomial using Sci-Kit Learn
polynomial_features = PolynomialFeatures(degree=3)
x_poly = polynomial_features.fit_transform(train_x)
model = LinearRegression()
model.fit(x_poly, train_y)

### YOUR CODE HERE - Use model to predict output of validation set
y_poly_pred = model.predict(polynomial_features.fit_transform(valid_x))

### YOUR CODE HERE - Calculate the RMSE. Report and plot the curve on the validation set.
print('poly train mse: ', mean_squared_error(model.predict(x_poly), train_y))
print('poly valid mse: ', mean_squared_error(y_poly_pred, valid_y))

