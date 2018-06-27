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

# choose either 'mnist_fashion' or 'mnist'
dataset = 'mnist'
model_name = '3_' + dataset + '_keras.h5'


def main():
    # create_dnn(batch_size=128, epochs=3)
    test_dnn()
    # print_nn_layer_outputs()


def create_dnn(batch_size=128, epochs=3):
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

    hist = model.fit(x_train, y_train,
                     batch_size=batch_size,
                     epochs=epochs,
                     verbose=1,
                     validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    print(hist.history)

    # save:
    #   the architecture of the model, allowing to re-create the model,
    #   the weights of the model,
    #   the training configuration (loss, optimizer),
    #   the state of the optimizer, allowing to resume training
    #   exactly where you left off
    model.save(model_name)


def test_dnn():
    (x_train, y_train), (x_test, y_test), _, _ = get_data()
    model = keras.models.load_model(model_name)

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
