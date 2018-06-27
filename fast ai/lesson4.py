# LESSON 4: STRUCTURED, TIME SERIES, & LANGUAGE MODELS
# http://course.fast.ai/lessons/lesson4.html

# Can enter eg
#   > learn
# into a cell to see info about the learner object

"""
Dropout:
    Reduce overfitting and dependance on individual neurons by randomly
    turning off some percentage 'p' of inputs during every run. Forces
    the activation to reduce reliance on individual pixels
    Keep changing which activations get dropped on each training run
    High p --> high generalisation, but will get low accuracy
    If p == 0.5:
        behind the scenes, the library will throw away half of the activations
        and double the remaining ones
    Turn on everything again when running the final network

Structured data:
    Continuous in data:
        Could be continuous or categorical in the model
    Categorical in data:
        Has to be categorical in data

    Something like a year or day of the week can be treated as a categorical
    variable (one-hot encoding) instead of a number
    Something encoded like 'a', 'b', 'c' will have to be treated as categorical
    Things with loads of levels (eg temp) should be treated as continuous
    Otherwise, you could try binning --> research

    Make something like:
        cat_vars = ['Store', 'DayOfWeek', 'Holiday', ... ]
        contin_vars = ['Distance', 'Temperature', ...]

Scaling in neural networks:
    Try to scale inputs to be roughly in the range of 0 -> 1 ish
    Could do something like
        data = (data - mean(data))/mean(data)
        or data = data - min(data)

Cardinality:
    Number of levels
    Eg cardinality of the number of days in the week is seven

Categorical Input Data:
    Use 'embedding matrices' eg using the following lookup table:
        Monday  | 1 | 2 | 0 | 1 |
        Tuesday | 2 | 5 | 3 | 9 |
        ...     | . | . | . | . |
        Sunday  | 0 | 1 | 9 | 0 |
        Unknown | 2 | 5 | 7 | 3 |
    Convert eg. 'Monday' into a rank 1 tensor of four floating point numbers
    Numbers are initialised randomly (or pretrained) and updated by backprop
    --> find the exact four numbers which work for Monday
    Then, if 'Monday' is the input, pass the four numbers in as four inputs

    To choose the embedding size, can make a rule like: min(50, (num+1)//2)
    > category_sizes = [['DaysOfWeek', 8], ['Category_i', 12], ...]
    > embedded_sizes = [(c, min(50, (c+1)//2)) for _, c in category_sizes]

    Using embedded matrices makes things like 'Sunday' a concept in 4D space
    instead of just a linear 1 or 0 input. So eg Saturday and Sunday would be
    close in the 4D space because they're weekends

    Note how 'DaysOfWeek' has 8 inputs = 7 days + an 8th for 'Unknown'

    Embeddings are suitable for any categorical variable, but not so much if
    the cardinality is too high (in which case the input should be continuous)

    Can THINK of the input as:
    (n x 1 one hot encoded array as input) * (embedded matrix array)
    --> just gets the row. Useful to think about, but libraries actually
        implement this as a lookup table

Natural Language Processing (NLP):
    Use embeddings for each input word (if doing things word by word)

    Can do fine tuning for NLP too - eg get a model which has learned the
    structure of English and can predict words, then replace the last couple
    layers with your own stuff which does sentiment analysis
    --> ie, first understand the language, then do other analysis

    A tokeniser will separate text into separate things - eg:
    don't --> do n't
    !!!!  --> ! ! ! !
    ...   --> ...
    and separate by space, etc
    spacy_tok() does this very well
    Then, each token gets its own embedding matrix. Common embedding sizes
    range from ~50 to ~600

    Keep word order (as opposed to 'bag of words' approach which doesn't)

    When understanding the language, there's no need to keep eg. movie reviews
    separate. Rather just feed in the entire dataset (in batches) and then
    worry about individual reviews when it comes to fine tuning/converting the
    model later on

    When fine tuning, start off by freezing previous layers as normal before
    unfreezing and training the entire network

    The fast.ai nlp library is super duper world class - consider using it
    --> see lesson 4

Gradient Clipping:
    Puts a limit to the maximum update amount (eg c = 0.3) to avoid overshoot

BPTT = Backprop Through Time
min_freq = minimum number of occurances of a word (otherwise labeled 'unknown')
"""
