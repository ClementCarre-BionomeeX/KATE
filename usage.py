import unittest

import tensorflow as tf

import src.layer.base as kate


class DenseLayerTester(kate.BaseLayerTester):
    layer_class = tf.keras.layers.Dense
    layer_args = {"units": 64, "activation": "relu"}
    input_shape = [None, 32]
    expected_shape = [None, 64]


if __name__ == "__main__":
    unittest.main()
