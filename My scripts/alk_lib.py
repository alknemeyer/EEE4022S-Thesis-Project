""" functions to help with quick debugging, visualistion, etc """
import numpy as np
import matplotlib.pyplot as plt
import os

from keras.datasets import mnist, fashion_mnist
from keras import backend as K
from keras.utils import to_categorical
import tensorflow as tf


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


def resave_as_pb(session, keep_var_names=None,
                 output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for
                         better portability.
    @return The frozen graph definition.
    """
    import tensorflow as tf
    from tensorflow.python.framework.graph_util\
        import convert_variables_to_constants

    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.namefor v in tf.global_variables())
                                .difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session,
                                                      input_graph_def,
                                                      output_names,
                                                      freeze_var_names)
        return frozen_graph


def save_model(model, dir_name):
    print(model.input)
    print(model.output)

    K.set_learning_phase(0)

    model.save(dir_name + '/model_complete.h5')

    frozen_graph = resave_as_pb(
        K.get_session(),
        output_names=[out.op.name for out in model.outputs])
    tf.train.write_graph(frozen_graph, dir_name,
                         'model_pb.pb', as_text=False)

    model_json = model.to_json()
    with open(dir_name + '/model_arch.json', 'w') as json_file:
        json_file.write(model_json)


def export_mvnc_graph(model, dir_name):
    # https://movidius.github.io/ncsdk/tools/compile.html

    frozen_graph = resave_as_pb(
        K.get_session(),
        output_names=[out.op.name for out in model.outputs])

    tf.train.write_graph(frozen_graph, dir_name,
                         'model_pb.pb', as_text=False)

    input_name = model.inputs[0].op.name
    output_name = model.outputs[0].op.name

    print('\n\nmvNCCompile model_pb.pb\
           -in=conv2d_3_input -on=dense_4/Softmax -s 12')

    command = 'mvNCCompile %s -in %s -on %s -o=%s'\
              % (dir_name + 'model_pb.pb', input_name, output_name, dir_name)

    print('Running the following command:\n%s' % command)
    os.system(command)

if __name__ == '__main__':
    pass
