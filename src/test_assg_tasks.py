import numpy as np
import pandas as pd
import sklearn
import random
# import unittest
from twisted.trial import unittest
from assg_tasks import load_onefeature_dataset
# from assg_tasks import add_dummy_feature
# from assg_tasks import predict
# from assg_tasks import loss_grad
# from assg_tasks import gradient_descent


class test_add_dummy_feature(unittest.TestCase):
    def setUp(self):
        pass

    def test_given_case(self):
        train_x, train_y = load_onefeature_dataset()
        train_x_dummy = add_dummy_feature(train_x)
        self.assertEqual(train_x.shape, (47, 1))
        self.assertEqual(train_x_dummy.shape, (47, 2))
        self.assertAlmostEqual(train_x_dummy[:, 0].sum(), 47.0)

    def test_random_case(self):
        train_x = np.random.random((2000, 24))
        train_x_dummy = add_dummy_feature(train_x)
        self.assertEqual(train_x.shape, (2000, 24))
        self.assertEqual(train_x_dummy.shape, (2000, 25))
        self.assertAlmostEqual(train_x_dummy[:, 0].sum(), 2000.0)


class test_predict(unittest.TestCase):
    def setUp(self):
        self.x, self.y_true = load_onefeature_dataset()
        self.x = add_dummy_feature(self.x)

    def test_given_case(self):
        theta = np.array([0.34, 0.10])
        y_pred = predict(self.x, theta)

        self.assertEqual(y_pred.shape, (47,))
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
        self.assertTrue(np.allclose(y_pred, expected_pred))

    def test_identity_case(self):
        # should result in predictions same as the single feature
        theta = np.array([0.0, 1.0])
        y_pred = predict(self.x, theta)

        self.assertEqual(y_pred.shape, (47,))
        self.assertTrue(np.allclose(y_pred, self.x[:, 1]))

    def test_random_case(self):
        # generate a known random x and theta using a seed.  Here we generate with 5 random features,
        # add in the dummy features, and then 6 random theta parameters.  Student code should already
        # handle more than the dummy and single feature if done as asked for here.
        np.random.seed(42)
        x = np.random.randn(50, 5)
        x = add_dummy_feature(x)
        theta = np.random.randn(6)
        y_pred = predict(x, theta)

        self.assertEqual(y_pred.shape, (50,))
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
        self.assertTrue(np.allclose(y_pred, expected_pred))


class test_loss_grad(unittest.TestCase):
    def setUp(self):
        self.x, self.y_true = load_onefeature_dataset()
        self.x = add_dummy_feature(self.x)

    def test_given_case(self):
        theta = np.array([0.0, 0.0])
        loss, gradients = loss_grad(self.x, theta, self.y_true)

        self.assertAlmostEqual(loss, 0.1311830962129149)
        self.assertEqual(gradients.shape, (2,))
        expected_gradients = np.array([-0.68082532, -0.21152827])
        self.assertTrue(np.allclose(gradients, expected_gradients))

    def test_optimal_case(self):
        # given theta parameters near optimal, would expect gradients to be small, close to 0
        theta = np.array([0.34, 0.10])
        loss, gradients = loss_grad(self.x, theta, self.y_true)

        self.assertAlmostEqual(loss, 0.004149661003713482)
        self.assertEqual(gradients.shape, (2,))
        expected_gradients = np.array([-0.00082532, -0.01152827])
        self.assertTrue(np.allclose(gradients, expected_gradients))

    def test_random_case(self):
        # generate a known random x and theta using a seed.  Here we generate with 5 random features,
        # add in the dummy features, and then 6 random theta parameters.  Student code should already
        # handle more than the dummy and single feature if done as asked for here.
        np.random.seed(42)
        x = np.random.randn(50, 5)
        y_true = np.random.randn(50)
        x = add_dummy_feature(x)
        theta = np.random.randn(6)
        loss, gradients = loss_grad(x, theta, y_true)

        self.assertAlmostEqual(loss, 2.7540470620966135)
        self.assertEqual(gradients.shape, (6,))
        expected_gradients = np.array(
            [-1.63612162, -0.7746676, 1.7010927, 0.75035861, 0.20324331, 1.0680429])
        self.assertTrue(np.allclose(gradients, expected_gradients))


class test_gradient_descent(unittest.TestCase):
    def setUp(self):
        self.x, self.y_true = load_onefeature_dataset()
        self.x = add_dummy_feature(self.x)

    def test_given_case(self):
        theta, history = gradient_descent(self.x, self.y_true)

        self.assertEqual(theta.shape, (2,))
        expected_theta = np.array([0.3404078,  0.10576262])
        self.assertTrue(np.allclose(theta, expected_theta))
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
        self.assertTrue(np.allclose(history, expected_history))
