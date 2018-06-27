# simple neural network to recognise handwritten digits
# https://www.tensorflow.org/get_started/mnist/beginners

# import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST data is split into three parts:
#   55,000 data points of training data (mnist.train)
#   10,000 points of test data (mnist.test)
#   5,000 points of validation data (mnist.validation).
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# the images are all 28 x 28 = 784 pixels
# 'None' means the dimension can be of any length
x = tf.placeholder(tf.float32, shape=[None, 784])

# the model: y = W*x + b
# 784 inputs, 10 outputs
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b

# the loss function: cross entropy = -sum(y_true * log(y_pred))
y_ = tf.placeholder(tf.float32, [None, 10])  # y_true
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# train using backpropagation with a learning rate of 0.5
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# this adds new operations to the graph which implement backpropagation and
# gradient descent. Then it gives you back a single operation which, when run,
# does a step of gradient descent training, slightly tweaking your variables
# to reduce the loss

# launch the model in an interactive session
sess = tf.InteractiveSession()

# create an operation to initialize the variables
tf.global_variables_initializer().run()

# run the training step 1000 times, in batches of 100 data points
# batches of small random data = stochastic training
# => stochastic gradient descent = SGD
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# evaluate the model
# argmax() gives the index of the highest entry in a tensor along some axis
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# correct_prediction is a list of booleans -> convert to a number
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# print the final accuracy
final_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images,
                                               y_: mnist.test.labels})
print('Accuracy = %d' % round(final_accuracy*100, 2))
