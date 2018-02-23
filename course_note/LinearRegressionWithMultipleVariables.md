# Week 2

### choosing features

Example: Housing price prediction
`price(x) = theta0 + theta1 * frontage + theta2 * depth `
This equation is got from the guess of the observation on the data. 


### Computing parameters analytically

- Normal equation: a method to solve the parameter `theta` analytically. It is achieved by initializing a cost function with `theta` as its parameters. Our goal is to minimize the cost function. For `cost(theta) = a*sqaure(theta) + b*theta + bias`, we get the lowest point by `partial_derivative(cost)/theta = 2*a*theta + b = 0`. In general, theta is a vector, therefore it is partial derivative on each dimension equals 0. 

Bellow is an implementation of how to get the normail equation.

```python 
import numpy as np
from numpy.linalg import inv

# input data
x = np.matrix([[1,2,3],[1,4,5],[1,4,4]])
y = np.array([5,6,7])
theta = inv(x.transpose()*x)*x*y.reshape([3,1])
```

- comparison:
|cost function    |advantage    |disadvatage    |
|-----------------|------------:|--------------:|
|normal equation  |* no need to iterate   | * require large memory |
|gradient descent |* works well even n is very large | * needs many iteration |

- summary: if the dataset contains too many features, then gradient descent is suitable. Else choose normal equation. 

### Normal Equation Noninvertibility
2 common situation to cause the noninvertable problem:
- redundant features. Some features are related.
- too many features. (use regularization)


