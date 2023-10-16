import os
import sys
import unittest

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import src.layer.base as kate


class TestKATE(unittest.TestCase):
    class DenseLayerTester(kate.BaseLayerTester):
        layer_class = tf.keras.layers.Dense
        layer_args = {"units": 64, "activation": "relu"}
        input_shape = [None, 32]
        expected_shape = [None, 64]
        known_values = {
            "input": [np.ones((32,))],
            "output": [
                np.ones((64,)) * 0.5
            ],  # This is a dummy value; real value depends on the layer's weights
        }

    def setUp(self):
        self.tester = self.DenseLayerTester()
        self.tester.setUp()

    def test_shapes(self):
        self.tester.test_shapes()

    def test_known_values(self):
        # As known values are "very", very very very likely to not be equals :)
        # we better test for faillure.
        with self.assertRaises(AssertionError):
            self.tester.test_known_values()

    def test_serialization(self):
        self.tester.test_serialization()


if __name__ == "__main__":
    unittest.main()
