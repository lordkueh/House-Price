# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 16:34:29 2018
Linear Regression and Multiple Linear Regression
@author: bubu
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import math

from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

#importing dataset

dataset = pd.read_csv(r'C:\Users\bubu\Downloads\holmusk\dataset_house_type_vs_price.csv')
dataset2 = pd.read_csv(r'C:\Users\bubu\Downloads\holmusk\house_age.csv')


X = dataset.iloc[:,0:1].values/10
y = dataset.iloc[:,4].values/10000


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)# Test size 0.3 fits better


# =============================================================================
# eliminating and replacing any outliers with mean value
# =============================================================================

mean = np.mean(X_train, axis=0)
y_mean = np.mean(y_train, axis=0)

X_median = np.median(X_train, axis=0)
y_median = np.median(y_train, axis=0)

sd = np.std(X_train, axis=0)

X_train_new = [X_median if element > 20 else element for element in X_train]
X_train = np.array(X_train_new)

y_train_new = [y_median if element > 800 else element for element in y_train]
y_train = np.array(y_train_new)

X_test_new = [X_median if element > 20 else element for element in X_test]
X_test = np.array(X_test_new)

y_test_new = [y_median if element > 800 else element for element in y_test]
y_test = np.array(y_test_new)


#==============================================================================

lm = linear_model.LinearRegression()
model = lm.fit(X_train,y_train)


predictions = lm.predict(X_test) #X_test

print(model.score(X_test,y_test))

regression_model_mse = mean_squared_error(predictions, y_test)

regression_model_error = math.sqrt(regression_model_mse)



plt.scatter(X, y, color = 'red')

plt.scatter(X_train, y_train, color = 'red')

plt.scatter(X_test, y_test, color = 'blue')

plt.plot(X_test, predictions, color='red',linewidth=3)

plt.title('House Price Prediction')
plt.xlabel('Age of House')
plt.ylabel('Resale Price')
plt.show()


