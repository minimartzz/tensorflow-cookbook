"""
# Implementing Unit Tests

`tf.test` module is used ensure more consistent code through testing and more efficient
debugging

Unit tests are used to check the functionality of the program. Tensorflow provides its
own test framework - create a custom layer that performs the unit tests

docs: https://www.tensorflow.org/api_docs/python/tf/test
"""
import tensorflow as tf
import numpy as np

class MyCustomGate(tf.keras.layers.Layer):
  def __init__(self, units, a1, b1):
    super(MyCustomGate, self).__init__()
    self.units = units
    self.a1 = a1
    self.b1 = b1
  
  def call(self, inputs):
    '''
    Compute f(x) = a1 * x + b1
    '''
    return inputs * self.a1 + self.b1

class MyCustomGateTest(tf.test.TestCase):
  def setUp(self):
    '''
    Unit test class.

    Setup is a hook method that is called before every test method
    '''
    super(MyCustomGateTest, self).setUp()
    # Configure the layer with 1 unit, a1 = 2 et b1=1
    self.my_custom_gate = MyCustomGate(1, 2, 1)
  
  def testMyCustomGateOutput(self):
    input_x = np.array([
      [1, 0, 0, 1],
      [1, 0, 0, 1]
    ])
    output = self.my_custom_gate(input_x)
    expected_output = np.array([[3, 1, 1, 3], [3, 1, 1, 3]])

    # Checks that the expected outputs all match the computed output
    self.assertAllEqual(output, expected_output)

tf.test.main()