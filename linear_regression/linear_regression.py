
import pickle
import pandas as pd
import numpy as np
import sympy
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
trainY = train_dataset.iloc[:,13] * 1000
testX = test_dataset.iloc[:,:13]
testY = test_dataset.iloc[:,13] * 1000
MSE = 10
theta0 = np.random.rand(1)
theta1 = np.random.rand(trainX.iloc[0].count())
count = 0
alpha = 0.001

# y_estimate = trainX.dot(theta1.transpose()) + theta0
# print(y_estimate)
# # update simultaneously
# # this is equivilant to theta0 - alpha * 1/m * sum(h(x(i)) - y(i))
# temp_theta0 = theta0 - alpha * np.sum(y_estimate - trainY) / trainY.count()
# # this is equivilant to theta1 - alpha * 1/m * sum((h(x(i)) - y(i)) * x(i))
# # I use * here instead of .dot because each column is multiplying the same value which is the corresponding h(x) - y
# temp_theta1 = theta1 - alpha * ((y_estimate - trainY) * trainX.transpose()).sum(axis=1)/trainY.count()
# theta0 = temp_theta0
# theta1 = temp_theta1
# # update MSE: 1/2 * 1/m * sum(square(h(x(i)) - y(i)))
# MSE = np.sum(np.square(theta0 + trainX.dot(theta1)-trainY)) / ( 2 * trainY.count())
# print('h(x):' + str(theta0 + trainX.dot(theta1)))
# print('y:' + str(trainY))
# print('MSE:' + str(MSE))




while MSE > 0.5:
    # h(x) = theta0 + theta1 * x
    y_estimate = trainX.dot(theta1.transpose()) + theta0
    # update simultaneously
    # this is equivilant to theta0 - alpha * 1/m * sum(h(x(i)) - y(i))
    temp_theta0 = theta0 - alpha * np.sum(y_estimate - trainY) / trainY.count()
    # this is equivilant to theta1 - alpha * 1/m * sum((h(x(i)) - y(i)) * x(i))
    # I use * here instead of .dot because each column is multiplying the same value which is the corresponding h(x) - y
    temp_theta1 = theta1 - alpha * ((y_estimate - trainY) * trainX.transpose()).sum(axis=1)/trainY.count()
    theta0 = temp_theta0
    theta1 = temp_theta1
    # update MSE: 1/2 * 1/m * sum(square(h(x(i)) - y(i)))
    MSE = np.sum(np.square(theta0 + trainX.dot(theta1)-trainY)) / ( 2 * trainY.count())
    print('How many iteration: ' + str(count))
    # print('h(x):' + str(theta0 + trainX.dot(theta1)))
    # print('y:' + str(trainY))
    print(MSE)
    count = count + 1
