'''
Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

# from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from alk_lib import save_model, export_mvnc_graph
import os

# choose either 'mnist_fashion' or 'mnist'
dataset = 'mnist'
model_name = '3_' + dataset + '_keras'

# disable tensorflow logging stuff (eg using AVX instructions)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    create_dnn(batch_size=128, epochs=1, train=False)

#    evaluate_models('3_mnist_keras', ['model_complete_with_dropout.h5',
#                                      'model_complete.h5'])
    # print_nn_layer_outputs()


def create_dnn(batch_size=128, epochs=3, train=True):
    (x_train, y_train), (x_test, y_test), input_shape, num_classes = get_data()

    # minute 12 of sirav "How To Deploy Keras Models to Production"
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

#    model.summary()

    if train is True:
        hist = model.fit(x_train, y_train,
                         batch_size=batch_size,
                         epochs=epochs,
                         verbose=1,
                         validation_data=(x_test, y_test))
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        print(hist.history)

        model.save('3_mnist_keras/model_complete_with_dropout.h5')
        model.save_weights('3_mnist_keras/model_weights.h5')

    # save:

    del model

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
#    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.load_weights('3_mnist_keras/model_weights.h5')

#    export_mvnc_graph(model, '3_mnist_keras')
#    save_model(model, '3_mnist_keras')

#    print('\n\nmvNCCompile model_pb.pb\
#           -in=conv2d_3_input -on=dense_4/Softmax -s 12')


def test_dnn(model_dir):
    (x_train, y_train), (x_test, y_test), _, _ = get_data()
    model = keras.models.load_model(model_dir)

    for i in range(5):
        pixels = x_test[i, :, :, 0]
        img_plot = plt.imshow(pixels)
        img_plot.set_cmap('Greys')
        plt.colorbar()

        pred = np.argmax(model.predict(x_test[i:i+1]))
        actual = np.argmax(y_test[i])

        if dataset == 'mnist':
            plt.title('Predicted: %d    Actual: %d' % (pred, actual))
        elif dataset == 'mnist_fashion':
            tmp = ['T-shirt/top',
                   'Trouser',
                   'Pullover',
                   'Dress',
                   'Coat',
                   'Sandal',
                   'Shirt',
                   'Sneaker',
                   'Bag',
                   'Ankle boot']
            plt.title('Predicted: ' + tmp[pred] + '    Actual: ' + tmp[actual])
        plt.show()


def evaluate_models(dir_name, list_of_model_names):
    (x_train, y_train), (x_test, y_test), _, _ = get_data()

    for model_name in list_of_model_names:
        model = keras.models.load_model(dir_name + '/' + model_name)
        score = model.evaluate(x_test, y_test, verbose=0)
        print('%s: test accuracy: %f' % (model_name, score[1]))


def get_data(verbose=0):
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
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test), input_shape, num_classes


def print_nn_layer_outputs():
    (x_train, y_train), (x_test, y_test), _, _ = get_data()
    model = keras.models.load_model(model_name)

    x = x_test[1:2]
    for layer_i in range(1, 8):
        # build a Keras function that returns the output of a certain layer
        # [x, 0] represents [input_data, test_mode]
        get_ith_layer_output = K.function([model.layers[0].input,
                                           K.learning_phase()],
                                          [model.layers[layer_i].output])
        layer_output = get_ith_layer_output([x, 0])[0]

        if len(layer_output.shape) == 4:
            layer_output = layer_output[0, :, :, 0]
        img_plot = plt.imshow(layer_output)
        img_plot.set_cmap('Greys')
        plt.colorbar()
        plt.title('Layer %d' % layer_i)
        plt.show()


if __name__ == '__main__':
    main()
