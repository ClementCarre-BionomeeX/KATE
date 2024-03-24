import unittest

import tensorflow as tf

import KATE


class DenseLayerTester(KATE.BaseLayerTester):
    layer_class = tf.keras.layers.Dense
    layer_args = {"units": 64, "activation": "relu"}
    input_shape = [None, 32]
    expected_shape = [None, 64]


class LambdaLayerTester(KATE.BaseLayerTester):
    layer_class = tf.keras.layers.Lambda
    layer_args = {"function": lambda x: x + 1}
    input_shape = [None, 2, 5]
    expected_shape = [None, 2, 5]
    known_values = {"input": [tf.ones((2, 5))], "output": [tf.ones((2, 5)) * 2]}


if __name__ == "__main__":
    unittest.main()
