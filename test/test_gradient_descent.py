import pytest
import numpy as np
from assg.tasks import load_onefeature_dataset
from assg.tasks import add_dummy_feature
from assg.tasks import gradient_descent


@pytest.fixture
def data():
    x, y_true = load_onefeature_dataset()
    x = add_dummy_feature(x)
    return x, y_true

def test_given_case(data):
    x, y_true = data
    theta, history = gradient_descent(x, y_true)
    assert theta.shape == (2,)

    expected_theta = np.array([0.3404078,  0.10576262])
    assert np.allclose(theta, expected_theta)

    expected_history = np.array(
        [0.1311831, 0.08543904, 0.05616284, 0.03742607, 0.02543454,
            0.01775996, 0.01284823, 0.00970472, 0.00769288, 0.0064053,
            0.00558125, 0.00505385, 0.00471632, 0.0045003, 0.00436205,
            0.00427357, 0.00421694, 0.0041807, 0.0041575, 0.00414266,
            0.00413316, 0.00412708, 0.00412318, 0.00412069, 0.0041191,
            0.00411808, 0.00411743, 0.00411701, 0.00411674, 0.00411657,
            0.00411646, 0.00411639, 0.00411635, 0.00411632, 0.0041163,
            0.00411629, 0.00411628, 0.00411627, 0.00411627, 0.00411627,
            0.00411627, 0.00411627, 0.00411627, 0.00411627, 0.00411627,
            0.00411627, 0.00411627, 0.00411627, 0.00411627, 0.00411627]
    )
    assert np.allclose(history, expected_history)
