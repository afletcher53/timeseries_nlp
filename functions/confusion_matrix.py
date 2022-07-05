import pickle

import numpy
from constants import (
    X_TEST_MULTI_INPUT_SAVE_FILE,
    X_TRAIN_MULTI_INPUT_SAVE_FILE,
    Y_TEST_MULTI_INPUT_SAVE_FILE,
    Y_TRAIN_MULTI_INPUT_SAVE_FILE,
)

import tensorflow as tf


def confusion_matrix(model, poison: bool = False):
    """Creates a confusion matrix, which can be poisoned (i.e. training data of
    one class provided to form confusion matrix)

    Args:
        model (model): Tensorflow model
        poison (bool, optional): Use training data from one class. Defaults to False.
    """
    if poison:
        x_test, y_test = poison_predictions(revisit=False)
    else:
        with open(X_TEST_MULTI_INPUT_SAVE_FILE, "rb") as f:
            x_test = pickle.load(f)
        with open(Y_TEST_MULTI_INPUT_SAVE_FILE, "rb") as f:
            y_test = pickle.load(f)

    y_pred = model.predict(x_test)
    y_pred_classes = numpy.where(
        y_pred > 0.5, 1, 0
    ).ravel()  # Predict class labels with weighting.
    cm = tf.math.confusion_matrix(
        labels=y_test, predictions=y_pred_classes, num_classes=2
    )
    print(cm)


def poison_predictions(revisit: bool = True):
    """Poisons test data to ensure that only one class is represented (to see if the model
       only predicts one class etc). Uses data that the model has trained on so should
       achieve greater accuracy.

    Args:
        revisit (bool, optional): Selection of class. Defaults to True.

    Returns:
        Numpy: x_test and y_test data
    """
    with open(X_TRAIN_MULTI_INPUT_SAVE_FILE, "rb") as f:
        x_test = pickle.load(f)
    with open(Y_TRAIN_MULTI_INPUT_SAVE_FILE, "rb") as f:
        y_test = pickle.load(f)
    y = numpy.where(y_test == int(revisit))[0]
    a = list()
    for _, element in enumerate(y):
        a.append(x_test[element])
    a = numpy.array(a)
    y = [int(revisit)] * a.shape[0]
    y = numpy.array(y)
    return a, y
