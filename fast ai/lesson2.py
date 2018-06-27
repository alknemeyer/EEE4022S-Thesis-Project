# LESSON 2: CONVOLUTIONAL NEURAL NETWORKS
# http://course.fast.ai/lessons/lesson2.html

# TODO: understand 'precompute' = minute 29 ish of lesson2
# precompute=True line:
#  Makes the classifier take advantage of previous layers
#  ConvLearner.pretrained(architechture, data, precompute=True)

"""
Create a link to data on the notebook server for easy access:
  > from IPhython.display import FileLink
  > FileLink('data/redux/my_submission.csv')

Debugging technique: print a couple random samples from cases where
  the data were and weren't predicted correctly
  the model was confident and right/wrong
  the model was unconfident (p = 0.5)

  eg:
  > most_uncertain = np.argsort(np.abs(probs - 0.5))
  > plots_idx(most_uncertain[:n_view], probs[most_uncertain])

Annealing:
  Drop the learning rate as the model is fitted. Step-wise annealing is often
  done, but a better approach is:
  Cosine annealing - learning rate starts high, stays there for a bit, then
  rapidly ish decreases, then stays low for a while to finish things off

Non Square Images:
  GPUs only handle square images, so passing a rectangular image to a model
  will usually result in only the middle section of the image being used

  Can use TTA to pick eg 4 random augmented versions of an image and take
  the average of the result as the answer OR consider adding stuff to the
  edges to make it square

  For test times - can slide the image (fixed crop locations) could be good
  May or may not work

CUDA out of memory:
  Increase batch size/image size until the 'out of memroy' error pops up.
  Stop and decrease, then restart the kernal

Image size trick:
  Start training on smaller images to get things going quickly, then switch to
  larger versions of the same images later on. Helps avoid overfitting
  Most modern nn architechtures (eg fully convolutional) can handle arbitrary
  sizes of images

  BUT be carefull of retraining a neural network that has been trained on a
  different image size (eg imagenet usually trained on 224x224). Don't worry
  if the new dataset is quite different to the one the pretrained nn was
  trained on

Activation:
  A number. The feature in this location with this level of confidence and
  probablity
"""
