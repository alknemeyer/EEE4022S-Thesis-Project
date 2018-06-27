# LESSON 5: COLLABORATIVE FILTERING; INSIDE THE TRAINING LOOP
# http://course.fast.ai/lessons/lesson5.html

"""
Embeddings:::
Collaborative Filtering:
    Based on the previous history of a user/thing, predict future
    behaviour/ratings/etc based on incomplete data

    Each input will map to its own embedding
    --> embeddings too large: computationally expensive + overfitting
    --> embeddings too small: inaccurate predictions
    Start off with randomly initialized numbers and solve

    For IMDB example:
        Every userID and moveID has their own embedding
        The embeddings could represent stuff like, how much does the user
        like dialogue driven movies? CGI? action? etc
        Dot product predicts the movie rating

    Good idea to find embeddings using a small, lightweight model and then
    later transfer them to a more complex model to save time on training
    The lightweight model COULD solve any meaningless problem/dummy task as
    long as the resulting embeddings are useful

Collaborative Filtering with Neural Networks:
    Concatenate the embeddings for the userID and movieID
    --> pass through nn
    Another advantage is that the embedding length for the different inputs
    don't have to be the same size (eg new input 'genre' has diff embedding
    size). Then add dropout, etc

Bias:
    Each input as a an offset which simply gets added
    --> prediction = (user_embedding*movie_embedding) + user_bias + movie_bias

Broadcasting:
    Duplicate a vector so that it can be added to a matrix
    Helps avoid doing loops

PyTorch:
    Modules etc all written in modern python as regular object oriented
    programming. New classes (or 'modules' as they call it) are written with a
    specific structure to make sure they can be used as part of eg. a nn

    Adding an underscore after the function makes the thing happen in place
    --> f_(b) vs b = f(b)

Momentum:
    TODO DIDN'T REALLY UNDERSTAND ->> see roughly 2 hours into lesson 5

    The gradient descent update amounts stay roughly the same after each step
    Ie, the update amount has momentum which changes slightly after each step
    Makes use of linear interpolation
    --> also gives more parameters to tune
    momentum = p = 0.98
    avg_loss = avg_loss * p + new_loss * (1 - p)
    --> this is an exponentially weighted moving average

Adam:
    TODO DIDN'T REALLY UNDERSTAND ->> see roughly 2 hours into lesson 5

    Very fast, but has historically tended to not give fantastic final results
    -------> subject to change!
    Use linear interpolation + momentum
    --> gets momentum of gradient + momentum of gradient squared
    Increases learning rate if variance is low
    Decreases learning rate if variance is high
"""
