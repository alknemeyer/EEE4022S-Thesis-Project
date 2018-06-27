# LESSON 3: IMPROVING YOUR IMAGE CLASSIFIER
# http://course.fast.ai/lessons/lesson3.html

# TODO: read up on: bn_freeze() = batch normalisation freeze = minute 20 ish

# wget http://files.fast.ai/part2/ etc etc rossman.tgz

"""
If te number of channels on the nn != number of channels of input data:
    input channels > input data, either:
        ignore extra channels
        ...
    input channels < input data, either:
        create combinations (eg averages) of input channels
        ...

Softmax:
    Activation function that outputs a number between 0 and 1
    Only exists in the last layer
    Sum of outputs of last layer = 1
    Algo:
        out_i = exp(in_i)/sum(exp.(all_inputs))
        where exp() is just used to make nums non-negative.
        it also magnifies differences between input numbers.
    Will tend to output one strong probability
    +: single-label classification (eg. is it a pic of a cat OR a dog)
    -: multi-label classification (eg. pic contains a river AND some trees)
    So, for multi-label, change the algo as follows:
        out_i = exp(in_i)/(1 + exp(in_i)))

    Don't use for things like regression (predicting values instead of labels)

Multi-label classification:
    Can't use regular approach of storing data (one folder per label)
    Instead, have to use .csv with labels approach

Dataset vs dataloader:
    Concept from PyTorch
    Dataset gives a single image
    Dataloader gives a (transformed) minibatch

data.resize():
    Loops through images and resizes (only done as a speedup)

learn.summary():
    tells you about the neural network structure (input/output shape, whether
    it's trainable, etc)

Structured data vs unstructured:
    unstructured
        Stuff like images with pixels
    structured
        Stuff like time series data, where a datapoint being in a certain
        column is meaningful
        Generally shared as .csv files

Filter:
    Matrix which gets element-wise multiplied by input image, with the output
    matrix being summed into a single number
    If the input + filter match, the output number will be big
    If they don't, the output number won't be big

CNN intro (intuition):
    https://www.youtube.com/watch?v=2-Ol7ZB0MmU
"""
