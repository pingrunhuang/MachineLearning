import tensorflow as tf
import numpy as np

# suppose we want to predic a equation with y = x^2-0.5
# create 300 point from -1 to 1 and shape it into 300 * 1 
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
# noise point with mean 0 and square error 0.05
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# each layer in a neural net should contain the following 4 parameters
def add_layer(inputs, in_size, out_size, activation_function=None):
	weights = tf.Variable(tf.random_normal([in_size, out_size]))
	# in case it is 0
	biases = tf.Variable(tf.random_normal([1,out_size]) + 0.1)
	hx = tf.matmul(inputs, weights) + biases

	if activation_function is None:
		output = hx
	else:
		output = activation_function(hx)

	return output

# add a hidden layer with 20 neuron
hidden1 = add_layer(xs, 1, 20, activation_function=tf.nn.relu)

# output layer with only 1 neuron
prediction = add_layer(hidden1, 20, 1)

# define cost function: 
cost = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))

# define gradient descent step with 0.1
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

# start training
# always remember to initialize the variables
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
iter_num = 1000

for i in range(iter_num):
	session.run(train_step, feed_dict={xs:x_data, ys:y_data})
	# print out every 50 times of iteration
	if i % 50 == 0:
		print(session.run(cost, feed_dict={xs: x_data, ys:y_data}))
