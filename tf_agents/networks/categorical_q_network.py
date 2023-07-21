# coding=utf-8
# Copyright 2020 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A Q-network for categorical DQN.

See "A Distributional Perspective on Reinforcement Learning" by Bellemare,
Dabney, and Munos (2017). https://arxiv.org/abs/1707.06887
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.networks import network
from tf_agents.networks import q_network
from tf_agents.specs import tensor_spec


@gin.configurable
class CategoricalQNetwork(network.Network):
  """Creates a categorical Q-network.

  It can be used to take an input of batched observations and outputs
  ([batch_size, num_actions, num_atoms], network's state).

  The first element of the output is a batch of logits based on the distribution
  called C51 from Bellemare et al., 2017 (https://arxiv.org/abs/1707.06887). The
  logits are used to compute approximate probability distributions for Q-values
  for each potential action, by computing the probabilities at the 51 points
  (called atoms) in np.linspace(-10.0, 10.0, 51).
  """

  def __init__(self,
               input_tensor_spec,
               action_spec,
               num_atoms=51,
               preprocessing_layers=None,
               preprocessing_combiner=None,
               conv_layer_params=None,
               fc_layer_params=None,
               activation_fn=tf.nn.relu,
               name='CategoricalQNetwork'):
    """Creates an instance of `CategoricalQNetwork`.

    The logits output by __call__ will ultimately have a shape of
    `[batch_size, num_actions, num_atoms]`, where `num_actions` is computed as
    `action_spec.maximum - action_spec.minimum + 1`. Each value is a logit for
    a particular action at a particular atom (see above).

    As an example, if
    `action_spec = tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 4)` and
    `num_atoms = 51`, the logits will have a shape of `[batch_size, 5, 51]`.

    Args:
      input_tensor_spec: A `tensor_spec.TensorSpec` specifying the observation
        spec.
      action_spec: A `tensor_spec.BoundedTensorSpec` representing the actions.
      num_atoms: The number of atoms to use in our approximate probability
        distributions. Defaults to 51 to produce C51.
      preprocessing_layers: (Optional.) A nest of `tf.keras.layers.Layer`
        representing preprocessing for the different observations.
        All of these layers must not be already built. For more details see
        the documentation of `networks.EncodingNetwork`.
      preprocessing_combiner: (Optional.) A keras layer that takes a flat list
        of tensors and combines them. Good options include
        `tf.keras.layers.Add` and `tf.keras.layers.Concatenate(axis=-1)`.
        This layer must not be already built. For more details see
        the documentation of `networks.EncodingNetwork`.
      conv_layer_params: Optional list of convolution layer parameters for
        observations, where each item is a length-three tuple indicating
        (num_units, kernel_size, stride).
      fc_layer_params: Optional list of fully connected parameters for
        observations, where each item is the number of units in the layer.
      activation_fn: Activation function, e.g. tf.nn.relu or tf.nn.leaky_relu.
      name: A string representing the name of the network.

    Raises:
      TypeError: `action_spec` is not a `BoundedTensorSpec`.
    """
    super(CategoricalQNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=(),
        name=name)

    if not isinstance(action_spec, tensor_spec.BoundedTensorSpec):
      raise TypeError('action_spec must be a BoundedTensorSpec. Got: %s' % (
          action_spec,))

    self._num_actions = action_spec.maximum - action_spec.minimum + 1
    self._num_atoms = num_atoms

    q_network_action_spec = tensor_spec.BoundedTensorSpec(
        (), tf.int32, minimum=0, maximum=self._num_actions * num_atoms - 1)

    self._q_network = q_network.QNetwork(
        input_tensor_spec=input_tensor_spec,
        action_spec=q_network_action_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        conv_layer_params=conv_layer_params,
        fc_layer_params=fc_layer_params,
        activation_fn=activation_fn,
        name=name)

  @property
  def num_atoms(self):
    return self._num_atoms

  def call(self, observation, step_type=None, network_state=(), training=False):
    """Runs the given observation through the network.

    Args:
      observation: The observation to provide to the network.
      step_type: The step type for the given observation. See `StepType` in
        time_step.py.
      network_state: A state tuple to pass to the network, mainly used by RNNs.
      training: Whether the output will be used for training.

    Returns:
      A tuple `(logits, network_state)`.
    """
    logits, network_state = self._q_network(
        observation, step_type, network_state, training=training)
    logits = tf.reshape(logits, [-1, self._num_actions, self._num_atoms])
    return logits, network_state
