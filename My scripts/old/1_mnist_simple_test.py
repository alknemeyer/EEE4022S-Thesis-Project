import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# get MNIST data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# the images are all 28 x 28 = 784 pixels
# 'None' means the dimension can be of any length
x = tf.placeholder(tf.float32, shape=[None, 784])

# the model: y = W*x + b
# 784 inputs, 10 outputs
W = tf.Variable(tf.zeros([784, 100]))
b = tf.Variable(tf.zeros([100]))
ytemp = tf.matmul(x, W) + b

W2 = tf.Variable(tf.zeros([100, 10]))
b2 = tf.Variable(tf.zeros([10]))

y = tf.matmul(ytemp, W2) + b2

# the loss function: cross entropy = -sum(y_true * log(y_pred))
y_ = tf.placeholder(tf.float32, [None, 10])  # y_true
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# train using backpropagation with a learning rate of 0.5
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# launch the model in an interactive session
sess = tf.InteractiveSession()

# create an operation to initialize the variables
tf.global_variables_initializer().run()

# run the training step 1000 times, in batches of 100 data points
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
