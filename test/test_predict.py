import pytest
import numpy as np
from assg.tasks import load_onefeature_dataset
from assg.tasks import add_dummy_feature
from assg.tasks import predict


@pytest.fixture
def data():
    x, y_true = load_onefeature_dataset()
    x = add_dummy_feature(x)
    return x, y_true

def test_given_case(data):
    x, y_true = data
    theta = np.array([0.34, 0.10])
    y_pred = predict(x, theta)

    assert y_pred.shape == (47,)

    expected_pred = np.array(
        [0.35314154, 0.28903593, 0.39079087, 0.26563229, 0.46710707,
            0.33800549, 0.28064115, 0.26703142, 0.26105332, 0.2755534,
            0.33228178, 0.3399134, 0.3259221, 0.65509933, 0.24680763,
            0.3780715, 0.2534217, 0.24273743, 0.41737435, 0.47105008,
            0.31027727, 0.32566771, 0.2895447, 0.33508004, 0.58030944,
            0.22543909, 0.27097443, 0.40681727, 0.36535213, 0.42093577,
            0.31943522, 0.21271973, 0.34500115, 0.48453261, 0.3158738,
            0.26830336, 0.24311901, 0.35670297, 0.62164739, 0.36051878,
            0.29717633, 0.37018549, 0.41203221, 0.23815846, 0.19389506,
            0.32108874, 0.23854004]
    )
    assert np.allclose(y_pred, expected_pred)

def test_identity_case(data):
    x, y_true = data

    # should result in predictions same as the single feature
    theta = np.array([0.0, 1.0])
    y_pred = predict(x, theta)

    assert y_pred.shape == (47,)
    assert np.allclose(y_pred, x[:, 1])

def test_random_case():
    # generate a known random x and theta using a seed.  Here we generate with 5 random features,
    # add in the dummy features, and then 6 random theta parameters.  Student code should already
    # handle more than the dummy and single feature if done as asked for here.
    np.random.seed(42)
    x = np.random.randn(50, 5)
    x = add_dummy_feature(x)
    theta = np.random.randn(6)
    y_pred = predict(x, theta)

    assert y_pred.shape == (50,)

    expected_pred = np.array(
        [-2.63033224,  3.11847582,  1.31747346, -1.53841258,  2.1032955,
            -2.15992489,  3.3124706, -2.03882096,  0.83463248, -1.4760662,
            -3.90813049, -3.48200297, -1.8129218,  0.63018334,  1.18908927,
            0.19255187,  2.00015485,  1.11698289,  0.84670253, -1.6004188,
            -2.50945879,  2.9377405, -6.66556037, -4.36487065,  0.82940254,
            -1.84471085, -3.90981482, -2.47296848, -0.34418788, -5.46843913,
            -1.49264575,  4.02422616,  1.49143951,  3.55872494, -4.54053627,
            0.11158461, -4.23613503,  2.34369435,  2.1766113, -2.15915891,
            0.44179985, -3.14639829,  1.82089587, -1.75101724,  0.28840202,
            -0.47997285, -1.47299819, -4.19147388, -2.44459343, -6.12884187]
    )
    assert np.allclose(y_pred, expected_pred)
