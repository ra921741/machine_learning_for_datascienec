# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 20:48:26 2019

@author: Bobby
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le_x = LabelEncoder()
x[:,0]=le_x.fit_transform(x[:,0])
oht = OneHotEncoder(categorical_features = [0])
x = oht.fit_transform(x).toarray()

le_y = LabelEncoder()
y=le_y.fit_transform(y)


from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

