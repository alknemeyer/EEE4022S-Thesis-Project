# https://www.tensorflow.org/get_started/mnist/pros

# saving and loading isn't really working, so after some internet research
# I switched to keras (which has a variety of advantages over tensorflow)

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt


def main():
    x, y_conv, keep_prob = create_dnn()
    train_dnn(x, y_conv, keep_prob, num_steps=1000,
              save_model=True, plot_accuracy=True)


def create_dnn(width=28, height=28):
    # 28 x 28 image
    # --> 32 * (14 x 14)
    # --> 64 * (7 x 7)

    # create placeholder for input x
    # each image is 28 x 28 = 784 pixels
    # 'None' corresponds to the batch size being unlimited
    x = tf.placeholder(tf.float32, shape=[None, width*height], name='input')

    # reshape x to a 4d tensor, with the second and third dimensions
    # corresponding to image width and height, and the final dimension
    # corresponding to the number of color channels
    x_image = tf.reshape(x, [-1, width, height, 1], name='x_image')

    # FIRST LAYER - convolution + max pooling

    # 5 x 5     patch size
    # 1         number of input channels
    # 32        number of output channels
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])  # one component for each output channel

    # next, convolve x_image with the weight tensor, add the bias, apply the
    # ReLU function, and finally max pool. The max_pool_2x2 method will reduce
    # the image size to 14x14 (ie downsample by 2x)
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # SECOND LAYER
    # maps 32 feature maps to 64 features for each 5x5 patch
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # densely connected layer
    # Now that the image size has been reduced to 7x7, we add a
    # fully-connected layer with 1024 neurons to allow processing on the
    # entire image. We reshape the tensor from the pooling layer into a batch
    # of vectors, multiply by a weight matrix, add a bias, and apply a ReLU
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # To reduce overfitting, we will apply dropout before the readout layer. We
    # create a placeholder for the probability that a neuron's output is kept
    # during dropout. This allows us to turn dropout on during training, and
    # turn it off during testing. TensorFlow's tf.nn.dropout op automatically
    # handles scaling neuron outputs in addition to masking them, so dropout
    # just works without any additional scaling
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # readout layer (like the mnist_simple script)
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return (x, y_conv, keep_prob)


def train_dnn(x, y_conv, keep_prob, num_steps=20000,
              save_model=False, plot_accuracy=False):
    # load MNIST data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # output = tf.nn.softmax(y_conv, name='output')  # TODO add this??

    # create placeholder for output y_
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_')

    # train using ADAM optimiser, dropout and log every 100th iteration
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1),
                                  name='correct_prediction')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # keep track of how the accuracy improves over time
    accuracy_history = []

    # create a session under 'with' so that it gets destroyed afterwards
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(num_steps):
            batch = mnist.train.next_batch(50)
            train_step.run(feed_dict={x: batch[0],
                                      y_: batch[1], keep_prob: 0.5})

            if i % 50 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
                accuracy_history.append(train_accuracy)

        test_accuracy = 100 * accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
        print('test accuracy %g' % round(test_accuracy, 2) + '%')

        if plot_accuracy is True:
            # plot the accuracy over time
            plt.plot(accuracy_history)
            plt.show()

        if save_model is True:
            # Save the variables to disk.
            path = './My scripts/2_mnist_advanced_model/2_mnist_advanced_model'
            save_path = saver.save(sess, path, global_step=num_steps)
            print("Model saved in file: %s" % save_path)

            return save_path
        return
    return


def weight_variable(shape):
    ''' initialize weights '''
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    ''' initialize bias variable '''
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    ''' convolution using a stride of one + zero padded so that the output is
        the same size as the input '''
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    ''' max pooling over 2x2 blocks '''
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


if __name__ == '__main__':
    main()
