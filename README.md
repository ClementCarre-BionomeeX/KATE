
# KATE: Keras Automated Testing Engine

KATE is a tool designed to streamline the process of testing Keras layers. Leveraging Python's `unittest` module, KATE provides a base test class that automates many of the common test scenarios for Keras layers. This allows developers to ensure that their custom Keras layers work as expected, both in terms of output shape and behavior, without having to rewrite the same test logic for each layer.

## Why Use KATE?

- **Consistency**: Ensure that every Keras layer you develop is tested with a consistent set of basic tests.
- **Speed**: Reduce the time needed to write tests for Keras layers. Once set up, adding new tests for different layers becomes a breeze.
- **Reliability**: By using a standardized set of tests, you can reduce the risk of bugs slipping through due to missing or incomplete tests.

## How to Use KATE:

### 1. Define Your Tester Class:

Derive your tester class from `KATE.BaseLayerTester` and set the required attributes:

- `layer_class`: The Keras layer class you want to test.
- `layer_args`: A dictionary containing the arguments to initialize the `layer_class`.
- `input_shape`: The input shape for the layer. The first dimension should typically be `None` (indicating the batch size).
- `expected_shape`: The expected output shape for the layer.

### Example:

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

### 2. Run the Tests:

Simply execute your script, and `unittest` will run all the tests defined in the `BaseLayerTester` for your specific layer.

## ⚠️ Warnings:

- **Do Not Import `BaseLayerTester` Directly in Test Suites**: Due to the way `unittest` discovers and runs tests, importing `BaseLayerTester` directly in a test suite will cause `unittest` to try to run the base tests, including those that are not meant to be run directly. This can lead to test failures. Always subclass `BaseLayerTester` and only import and use the subclasses in your test suites.
- **Ensure Correct Scope**: Make sure that any test classes derived from `BaseLayerTester` are only in scope when you actually want to run them. Avoid having them in global scope in modules that are imported elsewhere, as this can lead to unintentional test execution.

## Conclusion:

KATE provides an efficient way to ensure your Keras layers are robust and behave as expected. By abstracting away much of the repetitive testing logic, KATE allows you to focus on what matters most: developing high-quality Keras layers. Just remember the warnings about direct imports and scope to avoid any testing pitfalls. Happy testing!
