# -*- coding: utf-8 -*-
"""
Spyder Editor

Decision Tree Regression
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



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)# Test size 0.3 fits better

# Fitting decision tree regressor

from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train,y_train)



#Predicting new house price
#y_pred = regressor.predict([[300, 1]])

# Visualising decision tree regression (for higher resolution)

X_grid = np.arange(min(X_test), max(X_test), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('House Price Prediction')
plt.xlabel('Flat Model')
plt.ylabel('Resale Price')
plt.show()

predictions = regressor.predict(X_test)
regression_model_mse = mean_squared_error(predictions, y_test)

regression_model_error = math.sqrt(regression_model_mse)