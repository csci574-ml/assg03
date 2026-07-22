import math
import pathlib
import numpy as np
import pandas as pd
from assg_utils import PROJECT_ROOT


def load_onefeature_dataset():
    """Load in the dataset and return only the first feature for use for most of the
    assignment.

    Returns
    -------
    x : numpy array shape (47,1)
        The one feature values for training/fitting regression models with.  Has a single column feature.
    y_true : numpy array shape (47,)
        The true labels for training/fitting regression on.  A vector of 47 real valued target labels.
    """
    # get the data from file
    data = np.genfromtxt(PROJECT_ROOT / 'data' / 'data.csv', delimiter=',')

    # extract the features
    x = data[:, 0].reshape(-1, 1).copy()

    # get the true labels
    y_true = data[:, 4].copy()

    return x, y_true


def load_multifeature_dataset():
    """Load in the dataset, but this time return the full 4 features of the dataset for testing.

    Returns
    -------
    x : numpy array shape (47,4)
        The one feature values for training/fitting regression models with.  Has a single column feature.
    y_true : numpy array shape (47,)
        The true labels for training/fitting regression on.  A vector of 47 real valued target labels.
    """
    # get the data from file
    data = np.genfromtxt(PROJECT_ROOT / 'data' / 'data.csv', delimiter=',')

    # extract the features
    x = data[:, :4].copy()

    # get the true labels
    y_true = data[:, 4].copy()

    return x, y_true