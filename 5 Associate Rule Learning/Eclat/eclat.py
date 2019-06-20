#Eclat

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv',header=None)
n = len(dataset)
transactions =[]
for i in range(0,n):
    transaction1=[]
    m = len(dataset.values[i])
    for j in range(0,m):
        data=str(dataset.values[i,j])
        if data != 'nan':
            transaction1.append(data)
            transactions.append(transaction1)
            

#applying the eclat to dataset
from 