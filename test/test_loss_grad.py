import pytest
import numpy as np
from assg_tasks import load_onefeature_dataset
from assg_tasks import add_dummy_feature
from assg_tasks import loss_grad


@pytest.fixture
def data():
    x, y_true = load_onefeature_dataset()
    x = add_dummy_feature(x)
    return x, y_true

def test_given_case(data):
    x, y_true = data
    theta = np.array([0.0, 0.0])
    loss, gradients = loss_grad(x, theta, y_true)

    assert loss ==  pytest.approx(0.1311830962129149)
    assert gradients.shape == (2,)
    expected_gradients = np.array([-0.68082532, -0.21152827])
    assert np.allclose(gradients, expected_gradients)

def test_optimal_case(data):
    x, y_true = data
    # given theta parameters near optimal, would expect gradients to be small, close to 0
    theta = np.array([0.34, 0.10])
    loss, gradients = loss_grad(x, theta, y_true)

    assert loss == pytest.approx(0.004149661003713482)
    assert gradients.shape == (2,)
    expected_gradients = np.array([-0.00082532, -0.01152827])
    assert np.allclose(gradients, expected_gradients)

def test_random_case():
    # generate a known random x and theta using a seed.  Here we generate with 5 random features,
    # add in the dummy features, and then 6 random theta parameters.  Student code should already
    # handle more than the dummy and single feature if done as asked for here.
    np.random.seed(42)
    x = np.random.randn(50, 5)
    y_true = np.random.randn(50)
    x = add_dummy_feature(x)
    theta = np.random.randn(6)
    loss, gradients = loss_grad(x, theta, y_true)

    assert loss == pytest.approx(2.7540470620966135)
    assert gradients.shape == (6,)
    expected_gradients = np.array(
        [-1.63612162, -0.7746676, 1.7010927, 0.75035861, 0.20324331, 1.0680429])
    assert np.allclose(gradients, expected_gradients)
