'''
https://www.tensorflow.org/get_started/get_started
'''

import tensorflow as tf

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0, dtype=tf.float32)
print([node1, node2])

sess = tf.Session()

print(sess.run([node1, node2]))