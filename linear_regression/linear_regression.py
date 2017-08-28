
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

'''
 CRIM     per capita crime rate by town
 ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
 INDUS    proportion of non-retail business acres per town
 CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
 NOX      nitric oxides concentration (parts per 10 million)
 RM       average number of rooms per dwelling
 AGE      proportion of owner-occupied units built prior to 1940
 DIS      weighted distances to five Boston employment centres
 RAD      index of accessibility to radial highways
 TAX      full-value property-tax rate per $10,000
 PTRATIO  pupil-teacher ratio by town
 B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
 LSTAT    % lower status of the population
 MEDV     Median value of owner-occupied homes in $1000's
'''

df = pd.read_pickle('dataset/housing_dataset.pickle')

test_ratio = 0.3
train_dataset, test_dataset = train_test_split(df, test_size=test_ratio)

trainX = train_dataset.iloc[:,:13]
trainY = train_dataset.iloc[:,13]
testX = test_dataset.iloc[:,:13]
testY = test_dataset.iloc[:,13]

MSE = 10
theta0 = np.random.rand(1,1)
theta1 = np.random.rand(1,13)

# y = theta0 + theta1 * x
print(trainX.multiply(theta1,axis=0))


# while MSE > 0.5:
#     temp_theta0 = theta0 - trainX
