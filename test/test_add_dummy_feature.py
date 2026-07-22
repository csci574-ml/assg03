import pytest
import numpy as np
from assg_tasks import load_onefeature_dataset
from assg_tasks import add_dummy_feature

def test_given_case():
    train_x, train_y = load_onefeature_dataset()
    train_x_dummy = add_dummy_feature(train_x)
    assert train_x.shape == (47, 1)
    assert train_x_dummy.shape == (47, 2)
    assert train_x_dummy[:, 0].sum() == pytest.approx(47.0)

def test_random_case():
    train_x = np.random.random((2000, 24))
    train_x_dummy = add_dummy_feature(train_x)
    assert train_x.shape == (2000, 24)
    assert train_x_dummy.shape == (2000, 25)
    assert train_x_dummy[:, 0].sum() == pytest.approx(2000.0)