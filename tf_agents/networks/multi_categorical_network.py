"""Project inputs to multiple categorical distribution object."""

import gin
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.specs import tensor_spec
from tf_agents.specs import distribution_spec

class MultiCategoricalDistributionBlock(tfp.distributions.Blockwise):

  def __init__(self, logits, categories_shape):
    self.categories_shape = categories_shape
    distribs = self._create_distrib(logits)
    super().__init__(distributions = distribs)
    self._parameters['logits'] = logits

  def _create_distrib(self, logits):
    logits_splitted = tf.split(logits, self.categories_shape, -1)
    distribs = tf.nest.map_structure(
                    lambda l: tfp.distributions.Categorical(logits = l),
                    logits_splitted)
    return distribs

  def _mode(self):
    return self._flatten_and_concat_event(
      self._distribution.mode()
      )

@gin.configurable
class MultiCategoricalNetwork(network.DistributionNetwork):
  """Generates a set of tfp.distribution.Categorical by predicting logits"""
  def __init__(self,
            sample_spec,
            logits_init_output_factor=0.1,
            name='MultiCategoricalProjectionNetwork'):
    """Creates an instance of MultiCategoricalProjectionNetwork

    Args:
      sample_spec: A collection of `tensor_spec.BoundedTensorSpec` detailing
        the shape and dtypes of samples pulled from the output distribution.
      logits_init_output_factor: Output factor for initializing kernal
        logits weights.
      name: A string representing the name of the network.
    """

    categories_shape = self._categories_shape(sample_spec)
    n_unique_categories = np.sum(categories_shape)
    output_spec = self._output_distribution_spec(
      [n_unique_categories],
      sample_spec,
      categories_shape,
      name)

    super(MultiCategoricalNetwork, self).__init__(
      input_tensor_spec=None,
      state_spec = (),
      output_spec=output_spec,
      name=name
    )

    self._sample_spec = sample_spec
    self.categories_shape = categories_shape
    self.n_unique_categories = n_unique_categories

    self._projection_layer = tf.keras.layers.Dense(
        self.n_unique_categories,
        kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=logits_init_output_factor),
        bias_initializer=tf.keras.initializers.Zeros(),
        name='logits'
        )

  def _categories_shape(self, sample_spec):
    def _get_n_categories(array_spec):
      if not tensor_spec.is_bounded(array_spec):
        raise ValueError(
          'sample_spec must be bounded. Got: %s.' % type(array_spec))

      if not tensor_spec.is_discrete(array_spec):
        raise ValueError('sample_spec must be discrete. Got: %s.' % array_spec)

      n_categories = array_spec.maximum - array_spec.minimum +1
      return n_categories

    flattened_spec = tf.nest.flatten(sample_spec)
    categories_shape = tf.nest.map_structure(
      lambda s: _get_n_categories(s),
      flattened_spec)

    if len(categories_shape) == 1:
      categories_shape = categories_shape[0]

    return categories_shape

  def _output_distribution_spec(self, output_shape, sample_spec, categories_shape, network_name):
    input_param_spec = {
        'logits':
            tensor_spec.TensorSpec(
                shape=output_shape,
                dtype=tf.float32,
                name=network_name + '_logits'
            )
    }
    return distribution_spec.DistributionSpec(
        MultiCategoricalDistributionBlock,
        input_param_spec,
        sample_spec=sample_spec,
        categories_shape = categories_shape
        )

  def call(self, inputs, outer_rank, training=False, mask=None):
    batch_squash = utils.BatchSquash(outer_rank)
    inputs = batch_squash.flatten(inputs)
    inputs = tf.cast(inputs, tf.float32)

    logits = self._projection_layer(inputs, training=training)
    logits = tf.reshape(logits, [-1] + [self.n_unique_categories])
    logits = batch_squash.unflatten(logits)
    if mask is not None:
      # assume mask is a flattened array for now

      # If the action spec says each action should be shaped (1,), add another
      # dimension so the final shape is (B, 1, A), where A is the number of
      # actions. This will make Categorical emit events shaped (B, 1) rather
      # than (B,). Using axis -2 to allow for (B, T, 1, A) shaped q_values.
      if mask.shape.rank < logits.shape.rank:
        mask = tf.expand_dims(mask, -2)

      # Overwrite the logits for invalid actions to a very large negative
      # number. We do not use -inf because it produces NaNs in many tfp
      # functions.
      almost_neg_inf = tf.constant(logits.dtype.min, dtype = logits.dtype)
      logits = tf.compat.v2.where(
        tf.cast(mask, tf.bool), logits, almost_neg_inf)

    return self.output_spec.build_distribution(logits= logits), ()