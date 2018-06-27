# LESSON 7: RESNETS FROM SCRATCH
# http://course.fast.ai/lessons/lesson7.html

"""
RNN:
    BPTT length is usually determined by the amount of memory in your GPU
    --> matrix size = len(BPTT) x len(embedding)

Field (from Torch Text):
    Description of how we want to prepare the data
    --> lowercase/uppercase, tokenize method, etc
    eg: data.Field(lower=True, tokenize=list)

Randomise Text Data:
    Can't shuffle batches (as is done with images) but can add small
    deviations to the specified BPTT length
    Each minibatch will have a constant length, except for the end of the epoch
    (unless bbpt % total_batch_len == 0). Hence,
    > if self.h.size(1) != bs: self.init_hidden(bs)

pytorch F.log_softmax():
    Can't handle rank three tensors so flatten tensors when passing them
    TODO: F.log_softmax() might have been fixed already

RNN cells:
    Don't tend to use tanh as they can still easily result in exploding
    gradients. Intead, GRU (Gated SOMETHING SOMETHING) cells are used.
    They use a neural network to decide how much of the current hidden state
    should be remembered vs updated
    TODO: maybe read an article about RNN cells?

Linear Interpolation:
    Every time you see something of the form:
        h[t] = a * h[t] + (1 - a) * h[t-1]
    it's probably a linear interpolation, with a being a parameter which
    determines weighting

Smaller Datasets:
    Often useful to work with smaller datasets (both number of images and
    image size) when researching etc to speed up train time etc
    --> eg CIFAR 10

Stride 2 Convolutions:
    Don't do a convolution on every single 3x3 section
    --> instead, do every second 3x3. (ie go right by two after convolution,
    then down by two at the end of the row). Has a similar effect to max
    pooling 2x2 sections at the same time (ie halves image size)
    --> can do any Stride x Convolution

Adaptive Max Pooling:
    Pooling size defined by the desired resolution to create.
    Eg on a 28x28 image, doing a 7x7 adaptive max pool would result in a 4x4
    max pool being done
    Done at the end of the nn

    Modern CNNs usually have their penultimate layer as a 1x1 adaptive max pool
    --> ie, find the single largest cell and use that as the activation

Padding on Convolutions:
    Sorts out the edge cases (eg how to apply kernals to edges, corners, etc)

Batch Normalisation:
    Regularisation technique
    Normalise data using the mean and std dev of each channel/filter:
    > (x - mean(x))/std_dev(x)
    Then create two trainable parameters a and m, and use them to easily scale
    or shift the data without having to relearn every weight
    > return (x - mean(x))/std_dev(x) * m + a
    --> this prevents exploding gradients etc in dnn

    The changing m and a keep changing, which essentially adds noise to the
    network. Noise in a nn has the effect of regularisation. This results in
    the possibility of using a higher learning rate/lower dropout

    In practice, either m and a XOR mean and std_dev are made to be moving
    averages (didn't hear which)

    Best place to put BatchNorm is after the relu/activation

    Each stride 2 BatchNorm layer decreases the matrix size by half, which
    stops the possibility of a very dnn. So, you can add increase the depth of
    the model by inserting BNN layers with a stride of 1 between the stride 2
    layers
    x = self.conv1(x)
    for l, l2 in zip(self.layers_stride_2, self.layers_stride_1):
        x = l(l2(x))
    x = F.adaptive_max_pool2d(x, 1)
    return F.log_softmax(self.out(x))

    TODO: understand BatchNorm = 1.29.00 ish of lesson 7

First Layer of CNN:
    Create a Conv layer with a large kernal (eg 10 * 5x5 filters) (up to maybe
    32 * 12x12 filters)
    --> bigger kernal helps to find interesting, richer features

Resnet:
    Trains very well even with looaaddss of layers (dnn)

    Inherits from BnLayer:
        class ResnetLayer(BnLayer):
            def forward(self, x):
                return x + super().forward(x)

    Ie, prediction = input + f(input)
    > y = x + f(x)
    or,
    > f(x) = y - x
    aka trying to fit a function to residual = error between y and x
    or, try to find a set of weights to reduce the error between y and x

    Full resnet block is usually:
        return x + forward(forward(x))

    TODO: understand resnets = roughly 1.53.00 of lesson7

Class Activation Maps (CAM):
    Used to find which parts of an image are important (eg which parts contain
    pixels which indicate a cat) (like a heatmap for the import bits)

    Essentially overlay output of final convolutional layer onto input pic
    --> might need to stretch/resize layer output

    Use product of (final conv layer) * (predictions) to create matrix

    Use a 'hook' to get the output of intermediate layers (eg final conv layer)
"""
