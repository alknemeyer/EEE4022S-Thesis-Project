# LESSON 1: RECOGNISING CATS AND DOGS
# http://course.fast.ai/lessons/lesson1.html

"""

A note on using pre-trained models:
  It's a good idea to explore the data it was trained on to better
  understand its limits and bias (eg vgg16 was trained on relatively
  zoomed in pics of one object at a time)

Jupyter tricks:
  Fix import errors by restarting kernal + clearing all outputs
  Start a line with:
    % to give jupyter a command
    ! to do a bash command
  Get info on function arguments using:
    shift+tab
    shift+tab twice brings up documentation
    shift+tab three times brings up a whole new window
    Can also do:
      ?<function>
    View source code using:
      ??<function?
  Tab completion works for completing function names

  Press 'h' to learn more jupyter shortcuts

Image storage:
  3D array = (width) (height) (colour) = eg 240 120 3
  aka a "rank 3 tensor"

Neural Networks:
  Need multiple sets of linear and nonlinear layers
  Can approximate any function to any precision (infinitely flexible)
  Have no local minima (only global minima)
  Deeplearning = neural network with multiple hidden (nonlinear) layers

Convolution:
  Multiply blocks of pixels (eg 3x3 is very common) by a 'kernal' (another
  block of pixels) to produce an output image which could be blurred, have
  edges highlighted, etc
  --> linear operation

Nonlinear:
  Allows us to create arbitrarily complex functions
  Used to use sigmoid, now use ReLU (y = max(x, 0))

Gradient Descent/Learning Rate:
  Follow derivative of error towards minima
  Eg: X_n+1 = X_n + dy/dX_n * alpha
    where alpha is the 'learning rate'
    --> too high and you overshoot
    --> too low and you never reach the mark
    Recently, reserachers came up with a reliable way to set the learning rate
      "Cyclical Learning Rates for Training Neural Networks", 2015
      in fast.ai library as learn.lr_find()
      The idea:
        Start with a tiny alpha, then keep doubling it until the accuracy
        starts decreasing. Then, go back and stick with the learning rate
        which gave the best improvement.
        Plotting learning rate vs. loss would give a skewed parabola
        Choose the highest learning rate which still results in an improving
        loss
  https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0

Epoch:
  Go through entire dataset once. Usually done in batches

Combining Convolutions:
  eg:
    layer 1 recognises edges, gradients...
    layer 2 recognises corners, curves...
    layer 3 recognises eyeballs...
    etc

  Generally speaking, the earlier layers (as we've seen) have more
  general-purpose features. Therefore we would expect them to need less
  fine-tuning for new datasets. For this reason we will use different learning
  rates for different layers: the first few layers will be at 1e-4, the middle
  layers at 1e-3, and our FC layers we'll leave at 1e-2 as before. fast.ai
  refers to this as differential learning rates, although there's no standard
  name for this techique in the literature that they're aware of

Data Augmentation:
  If you try training for more epochs, you'll notice that we start to overfit,
  which means that our model is learning to recognize the specific images in
  the training set, rather than generalizaing such that we also get good
  results on the validation set. One way to fix this is to effectively create
  more data, through data augmentation. This refers to randomly changing the
  images in ways that shouldn't impact their interpretation, such as horizontal
  flipping, zooming, and rotating

  There is something else we can do with data augmentation: use it at
  inference time (also known as test time). Not surprisingly, this is known as
  test time augmentation, or just TTA.

Stochastic Gradient Descent with Restarts (SGDR):
  Variant of learning rate annealing, which gradually decreases the learning
  rate as training progresses. This is helpful because as we get closer to the
  optimal weights, we want to take smaller steps.

  However, we may find ourselves in a part of the weight space that isn't very
  resilient - that is, small changes to the weights may result in big changes
  to the loss. We want to encourage our model to find parts of the weight
  space that are both accurate and stable. Therefore, from time to time we
  increase the learning rate (this is the 'restarts' in 'SGDR'), which will
  force the model to jump to a different part of the weight space if the
  current area is "spikey". In this paper they call it a "cyclic LR schedule"

  Frequency of restarts is set by cycl_len = number of epochs in a full cycle

  cycl_mult increases the length of a cycle (eg double the length each time
  from 1 epoch per cycle, then 2 epochs per cycle, then 4 epochs...)

Parameters:
  Learned by fitting a model to the data

Hyperameters:
  Another kind of parameter, that cannot be directly learned from the regular
  training process. These parameters express “higher-level” properties of the
  model such as its complexity or how fast it should learn. Two examples of
  hyperparameters are the learning rate and the number of epochs

Loss and Accuracy:
  Accuracy is the ratio of correct prediction to the total number of
  predictions

  In machine learning the loss function or cost function is representing the
  price paid for inaccuracy of predictions
  The loss associated with one example in binary classification is given by:
  -(y * log(p) + (1-y) * log (1-p)) where y is the true label of x and p is
  the probability predicted by our model that the label is 1

Confusion Matrix:
  Terminology:
    true positives (TP): predict yes, actual yes
    true negatives (TN): predict no, actual no
    false positives (FP): predict yes, actual no (aka "Type I error")
    false negatives (FN): predict no, actual yes (aka "Type II error")

    Accuracy = (TP + TN)/total
      overall, how often is classifier correct?
    Misclassification Rate = (FP + FN)/total
      overall, how often is it wrong?
      = 1 - accuracy
      aka "error rate"
    True Positive Rate = TP/actual yes
      When it's actually yes, how often does it predict yes?
      aka "sensitivity" or "recall"
    False Positive Rate = FP/actual no
      When it's actually no, how often does it predict yes?
    Specificity = TN/actual no
      When it's actually no, how often does it predict no?
      = 1 - False Positive Rate
    Precision = TP/predicted yes
      When it predicts yes, how often is it correct?
    Prevalence = actual yes/total
      How often does the yes condition actually occur in our sample?

Other Testing Terminology:
  Positive Predictive Value:
    Very similar to precision, except that it takes prevalence into account.
    In the case where the classes are perfectly balanced (meaning the
    prevalence is 50%), the positive predictive value (PPV) is equivalent to
    precision
  Null Error Rate:
    This is how often you would be wrong if you always predicted the
    majority class. This can be a useful baseline metric to compare your
    classifier against. However, the best classifier for a particular
    application will sometimes have a higher error rate than the null error
    rate, as demonstrated by the Accuracy Paradox.
  Cohen's Kappa:
    This is essentially a measure of how well the classifier performed as
    compared to how well it would have performed simply by chance. In other
    words, a model will have a high Kappa score if there is a big difference
    between the accuracy and the null error rate
  F Score:
    This is a weighted average of the true positive rate (recall) and
    precision
  ROC Curve:
    This is a commonly used graph that summarizes the performance of a
    classifier over all possible thresholds. It is generated by plotting the
    True Positive Rate (y-axis) against the False Positive Rate (x-axis) as
    you vary the threshold for assigning observations to a given class
"""
from sklearn.metrics import confusion_matrix
import matplotlib.pyplt as plt
import numpy as np
import itertools
y_actual = np.array([1, 0, 1, 1, 0])
y_predicted = np.array([1, 0, 1, 0, 1])
cm = confusion_matrix(y_actual, y_predicted)


def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    (This function is copied from the scikit docs.)
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


classes = 123  # data.classes
plot_confusion_matrix(cm, classes)


# OLD LESSON 1 NOTES (2013 VERSION OF THE COURSE)
"""
Order of software (from top to bottom):
  VGG16 (main DL library)
  Theano (takes python code and turns it into compiled GPU code)
  CUDA cuDNN (Deep Neural Network library from which Theano calls functions)

Convert between Theano backend to TensorFlow (optional):
    change "th" -> "tf" and "theano" to "tensorflow" in
    > nano ~/.keras/keras.json/
"""
# the workflow was basically:
# 1. create a regular feedforward neural network
# 2. replace the random initial weights with those from vgg16
# 3. delete the last layer of the nn and replace with two new neurons
# 4. train the final layer of the nn on new data, keeping the rest of it static

# number of images to train on simultaneously (GPU)
# not recommended to use more than 64, and older GPUs will handle less
batch_size = 64
