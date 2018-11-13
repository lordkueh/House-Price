# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 12:00:47 2018
Random Forest Regression
@author: bubu
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import math


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#importing dataset


dataset = pd.read_csv(r'C:\Users\bubu\Downloads\holmusk\dataset_house_type_vs_price.csv')
dataset2 = pd.read_csv(r'C:\Users\bubu\Downloads\holmusk\house_age.csv')
X = dataset2.iloc[:,0:1].values
y = dataset.iloc[:,4].values /1000


# =============================================================================
# #Splitting dataset to training and testing data
# 
# =============================================================================



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)# Test size 0.3 fits better


# Fitting decision tree regressor
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 250, random_state = 0)
regressor.fit(X_train,y_train)

#Predicting new house price
y_pred = regressor.predict(275)

# Visualising decision tree regression (for higher resolution)

X_grid = np.arange(min(X), max(X), 0.001)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('House Price Prediction (Random Forest)')
plt.xlabel('Flat Model')
plt.ylabel('Resale Price')
plt.show()


predictions = regressor.predict(X_test)
regression_model_mse = mean_squared_error(predictions, y_test)

regression_model_error = math.sqrt(regression_model_mse)