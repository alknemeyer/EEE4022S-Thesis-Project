# LESSON 6: INTERPRETING EMBEDDINGS; RNNS FROM SCRATCH
# http://course.fast.ai/lessons/lesson6.html

# TODO check out 'autoencoders'

"""
Embedding as Feature Engineering:
    Can use a neural network to find embeddings, and then use those embeddings
    in an entirely different model (such as random forest)
    Also just generally used to find similarities between any things
    --> combine with PCA for lower dimensionality understanding

Recurrent Neural Networks:
    RATHER JUST WATCH THE VID - ~1.30.00 of lesson6 onwards + lesson7
    --> also, maybe do some online reading

    Have a 'state' (useful for context)
    --> have memory
    Can work with variable length sequences

    input0 --> layer --> layer --> ... --> output
                /         /
            input1    input2    ...

    Can extend an eg. 8 layer rnn to more layers by predicting a 9th character,
    appending it to the input list and then calling again using something like
    input[1:] --> append again, call input[2:] --> repeat as long as you want

    TODO see video for various ways of handling input data (roughly 1.30.00)
    --> ie overlapping input sentences vs multi output model
    version 1: lots of overlap (relearning of the same thing)
    [1, 2, 3, 4] --> predict 5
    [2, 3, 4, 5] --> predict 6
    [3, 4, 5, 6] --> predict 7
    version 2: mulitple output
    [1, 2, 3, 4] --> predict [2, 3, 4, 5]
    [2, 3, 4, 5] --> predict [3, 4, 5, 6]

    See RNN.png

Vanishing/Exploding Gradients:
    Applying the same matrix multiple times (for each input) essentially could
    cause the numbers to increase/decrease massively. This can be prevented by
    making the initial hidden layer of a size that won't cause activations to
    increase or decrease
    --> choose the identity matrix as the initial value for the hidden weight
        matrix. As a side effect, the learning rate can also be higher
"""
