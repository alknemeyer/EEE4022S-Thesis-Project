""" functions to help with quick debugging, visualistion, etc """
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist, fashion_mnist
from keras import backend as K
from keras.utils import to_categorical


def plot_imag(pixels, label=None, width=28, height=28, plot_hist=False):
    # reshape to be 28 x 28
    pixels = np.array([pixels[i*width:(i+1)*width] for i in range(height)])

    # plot the image
    img_plot = plt.imshow(pixels)
    img_plot.set_cmap('Greys')
    plt.colorbar()
    plt.show()

    if plot_hist is True:
        # histogram of the colours in the pic
        plt.hist(pixels.flatten(), 256, range=(0.0, 1.0), fc='k', ec='k')
        plt.show()

    # for i in range(784):
    #    print(int(pixels[i] > 0.5), end='')
    #    if (i + 1) % 28 == 0:
    #       print()

    # val = 232 + round(pixel * 23)
    #    return '\x1b[48;5;{}m \x1b[0m'.format(int(val))


def get_mnist_data(dataset='mnist', verbose=0):
    """ get mnist or fashion mnist data + some meta data """
    # the data, shuffled and split between train and test sets
    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        img_rows, img_cols, num_classes = 28, 28, 10
    elif dataset == 'mnist_fashion':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        img_rows, img_cols, num_classes = 28, 28, 10
    else:
        print('Invalid dataset name')
        quit()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255

    if verbose:
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test), input_shape, num_classes


def binary_loss(y_true, y_pred):
    """ binary loss
    y_true: the actual label (1 or 0)
    y_pred: the predicted probabilty that the label is 1 """
    return -np.mean(y_true*np.log(y_pred) + (1 - y_true)*np.log(1 - y_pred))


def resave_as_tensorflow():
    print('"resave_as_tensorflow()" not implemented yet')

    # load model
    # K.set_learning_phase(0)
    # model = keras.models.load_model(model_name)


if __name__ == '__main__':
    pass
