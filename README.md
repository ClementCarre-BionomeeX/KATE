# KATE: Keras Automated Testing Engine

example usage:

```python
import unittest

import tensorflow as tf

import KATE


class DenseLayerTester(KATE.BaseLayerTester):
    layer_class = tf.keras.layers.Dense
    layer_args = {"units": 64, "activation": "relu"}
    input_shape = [None, 32]
    expected_shape = [None, 64]


if __name__ == "__main__":
    unittest.main()

```
