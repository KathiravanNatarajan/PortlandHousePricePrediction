# -*- coding: utf-8 -*-
"""
Created on Thu Feb 09 09:45:37 2017

@author: sunkathirav@gmail.com
"""

import numpy as np 
from sklearn import cross_validation, linear_model

import pandas as pd 

house = pd.read_csv('housing-prices-4-features-portland-or.csv')



X = house[:,0:4].as_matrix()
y = house.price.as_matrix()
print X

print X.shape
print y.shape

X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.4, random_state=0)

clf = linear_model.LinearRegression()

clf.fit(X_train,y_train)

print clf.predict(X_test,y_test)
