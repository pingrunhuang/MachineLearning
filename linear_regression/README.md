# Cost Function
Cost function is the measurement that I are going to evaluate our linear regression model. Suppose our target model is h(x) = theta(0) + theta(1)x, the cost function will look like the following:
J(theta(0), theta(1)) = 1/2 * 1/m * sum(square(h(x(i)) - y(i)))

which is also called Squared Error Function or Mean Squared Error(MSE)

Our goal is to get the appropriate theta(0) and theta(1) so the the value of J could be the smallest.

# Gradient descent

* should simultaneous update theta(0) and theta(1)
temp_theta0 = theta0 - alpha * np.diff(J(theta0, theta1), theta0) = theta0 - alpha * 1/2 * 1/m * sum(h(x(i) - y(i)))
temp_theta1 = theta1 - alpha * np.diff(J(theta0, theta1), theta1) = theta1 - alpha * 1/2 * 1/m * sum(h(x(i) - y(i))) * x(i)
theta0=temp_theta0
theta1=temp_theta1

* turns out that the gradient decent will always get the global optimal.

* different kind of gradient descent:
  * Batch gradient descent: 
