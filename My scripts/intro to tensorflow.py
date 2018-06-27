# first this:
# https://www.tensorflow.org/get_started/get_started

# canonical import:
import tensorflow as tf

# tensors
3  # a rank 0 tensor; a scalar with shape []
[1., 2., 3.]  # a rank 1 tensor; a vector with shape [3]
[[1., 2., 3.], [4., 5., 6.]]  # a rank 2 tensor; a matrix with shape [2, 3]
[[[1., 2., 3.]], [[7., 8., 9.]]]  # a rank 3 tensor with shape [2, 1, 3]

# tensorflow program overview:
# first you build the computational graph (nodes = operations, edges = data)
# then you run the graph
# --> so, nodes don't do anything until run in a session
sess = tf.Session()
node1 = tf.constant(3.0, dtype=tf.float32)  # constants
node2 = tf.constant(4.0)  # also tf.float32 implicitly
print(sess.run([node1, node2]))

# To do efficient numerical computing in Python, we typically use libraries
# like NumPy that do expensive operations such as matrix multiplication
# outside Python, using highly efficient code implemented in another language.
# Unfortunately, there can still be a lot of overhead from switching back to
# Python every operation. This overhead is especially bad if you want to run
# computations on GPUs or in a distributed manner, where there can be a high
# cost to transferring data.
# TensorFlow also does its heavy lifting outside Python, but it takes things a
# step further to avoid this overhead. Instead of running a single expensive
# operation independently from Python, TensorFlow lets us describe a graph of
# interacting operations that run entirely outside Python
import numpy as np

# worth noting: numpy typically makes and returns copies of things (instead
# of doing calcs in place). Exceptions are in cases like:
x = np.array()
x.sort()
# whereas the following would make a copy
x = np.array()
x = np.sort(x)
