# Cost function
We don't use the cost function as the linear regression use because as for the discrete problems, it turns out that the cost function is not a convex function which will lead to local optimal.  
Cost function of linear regression is defined by
```
J(theta) = sum(cost(h(x), y)) / length(y)

where

if y = 1:
  cost(h(x), y) = - log(h(x))
if y = 0:
  cost(h(x), y) = - log(1-h(x))

So to sum up:
cost(h(x), y) = -y * log(h(x)) - (1-y) * log(1-h(x))

# notice that h(x) here is
h(x) = 1 / (1 + exp-(theta * x))
instead of h(x) = theta * x in linear regression
```
The goal is to penalize the error prediction with large cost while the right prediction with 0 cost.

## How to minimize the cost function with gradient descent?
theta = theta - alpha * sum((h(x) - y) * x) / length(y)

## other algorithms for minimizing cost function
* conjugate descent
* BFGS
* L-BFGS
these function do not require manually setting alpha and more faster then gradient descent.
