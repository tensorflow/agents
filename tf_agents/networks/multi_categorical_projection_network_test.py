"""Tests for tf_agents.networks.multi_categorical_projection_network."""
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.networks import multi_categorical_projection_network
from tf_agents.specs import tensor_spec


def _get_inputs(batch_size, num_input_dims):
  return tf.random.uniform([batch_size, num_input_dims])

class MultiCategoricalProjectionNetworkTest(tf.test.TestCase):

  def testBuild(self):
    output_spec = [
      tensor_spec.BoundedTensorSpec([], tf.int32, 0, 1),
      tensor_spec.BoundedTensorSpec([], tf.int32, 0, 4)]
    network = multi_categorical_projection_network.MultiCategoricalProjectionNetwork(
      output_spec)
    inputs = _get_inputs(batch_size=3, num_input_dims=5)

    distribution, _ = network(inputs, outer_rank=1)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    sample = self.evaluate(distribution.sample())

    self.assertEqual(multi_categorical_projection_network.MultiCategoricalDistributionBlock,
      type(distribution))
    self.assertEqual(tfp.distributions.Categorical, type(distribution.distributions[0]))
    self.assertEqual(2, len(distribution.distributions))
    self.assertEqual((3, 7), distribution._parameters['logits'].shape)
    self.assertEqual((3, 2), sample.shape)

  def testTrainableVariables(self):
    output_spec = [
      tensor_spec.BoundedTensorSpec([], tf.int32, 0, 2),
      tensor_spec.BoundedTensorSpec([], tf.int32, 0, 5)]
    network = multi_categorical_projection_network.MultiCategoricalProjectionNetwork(
      output_spec)
    inputs = _get_inputs(batch_size=3, num_input_dims=5)

    network(inputs, outer_rank=1)
    self.evaluate(tf.compat.v1.global_variables_initializer())

    self.assertEqual(2, len(network.trainable_variables))
    self.assertEqual((5, 9), network.trainable_variables[0].shape)
    self.assertEqual((9,), network.trainable_variables[1].shape)


if __name__ == '__main__':
  tf.test.main()