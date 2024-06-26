import tempfile
import unittest

import numpy as np
import keras


class BaseLayerTester(unittest.TestCase):
    layer_class = None
    layer_args = {}
    input_shape = None
    expected_shape = None
    known_values = {"input": [], "output": []}

    def setUp(self):  # This method is automatically called before each test
        if not self.layer_class:
            raise NotImplementedError("Subclasses must define a layer_class attribute.")

        self.layer = self.layer_class(**self.layer_args)
        self.model = keras.models.Sequential()
        self.model.add(self.layer)
        self.model.build(self.input_shape)

    def test_summary(self):
        self.model.summary()

    def test_shapes(self):
        output_shape = self.model.predict(
            np.random.randn(1, *self.input_shape[1:])
        ).shape
        self.assertEqual(
            output_shape,
            (1, *self.expected_shape[1:]),
            f"Expected shape: {self.expected_shape}, but got: {output_shape}",
        )

    def test_known_values(self):
        if not self.known_values["input"] or not self.known_values["output"]:
            print("Skipping known values test due to lack of provided known values.")
            return

        for inp, expected_out in zip(
            self.known_values["input"], self.known_values["output"]
        ):
            predicted_out = self.model.predict(np.array([inp]))
            self.assertTrue(
                keras.ops.sum(keras.ops.abs(predicted_out - expected_out)) < 1e-3
            )

    def test_serialization(self):
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=True) as tmp:
            self.model.save(tmp.name)
            loaded_model = keras.models.load_model(
                tmp.name, custom_objects={self.layer_class.__name__: self.layer_class}
            )
