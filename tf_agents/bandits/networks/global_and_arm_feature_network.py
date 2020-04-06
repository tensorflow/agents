# coding=utf-8
# Copyright 2018 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Networks that take as input global and per-arm features, and output rewards."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.bandits.specs import utils as bandit_spec_utils
from tf_agents.networks import encoding_network
from tf_agents.networks import network
from tf_agents.networks import q_network
from tf_agents.specs import tensor_spec


def create_feed_forward_common_tower_network(observation_spec, global_layers,
                                             arm_layers, common_layers):
  """Creates a common tower network with feedforward towers.

  Args:
    observation_spec: A nested tensor spec containing the specs for global as
      well as per-arm observations.
    global_layers: Iterable of ints. Specifies the layers of the global tower.
    arm_layers: Iterable of ints. Specifies the layers of the arm tower.
    common_layers: Iterable of ints. Specifies the layers of the common tower.

  Returns:
    A network that takes observations adhering observation_spec and outputs
    reward estimates for every action.
  """
  global_network = encoding_network.EncodingNetwork(
      input_tensor_spec=observation_spec[bandit_spec_utils.GLOBAL_FEATURE_KEY],
      fc_layer_params=global_layers)

  one_dim_per_arm_obs = tensor_spec.TensorSpec(
      shape=observation_spec[bandit_spec_utils.PER_ARM_FEATURE_KEY].shape[1:],
      dtype=tf.float32)
  arm_network = encoding_network.EncodingNetwork(
      input_tensor_spec=one_dim_per_arm_obs,
      fc_layer_params=arm_layers)
  common_input_dim = global_layers[-1] + arm_layers[-1]
  common_input_spec = tensor_spec.TensorSpec(
      shape=(common_input_dim,), dtype=tf.float32)
  common_network = q_network.QNetwork(
      input_tensor_spec=common_input_spec,
      action_spec=tensor_spec.BoundedTensorSpec(
          shape=(), minimum=0, maximum=0, dtype=tf.int32),
      fc_layer_params=common_layers)
  return GlobalAndArmCommonTowerNetwork(observation_spec, global_network,
                                        arm_network, common_network)


def create_feed_forward_dot_product_network(observation_spec, global_layers,
                                            arm_layers):
  """Creates a dot product network with feedforward towers.

  Args:
    observation_spec: A nested tensor spec containing the specs for global as
      well as per-arm observations.
    global_layers: Iterable of ints. Specifies the layers of the global tower.
    arm_layers: Iterable of ints. Specifies the layers of the arm tower. The
      last element of arm_layers has to be equal to that of global_layers.

  Returns:
    A dot product network that takes observations adhering observation_spec and
    outputs reward estimates for every action.

  Raises:
    ValueError: If the last arm layer does not match the last global layer.
  """

  if arm_layers[-1] != global_layers[-1]:
    raise ValueError('Last layer size of global and arm layers should match.')

  global_network = encoding_network.EncodingNetwork(
      input_tensor_spec=observation_spec[bandit_spec_utils.GLOBAL_FEATURE_KEY],
      fc_layer_params=global_layers)
  one_dim_per_arm_obs = tensor_spec.TensorSpec(
      shape=observation_spec[bandit_spec_utils.PER_ARM_FEATURE_KEY].shape[1:],
      dtype=tf.float32)
  arm_network = encoding_network.EncodingNetwork(
      input_tensor_spec=one_dim_per_arm_obs,
      fc_layer_params=arm_layers)
  return GlobalAndArmDotProductNetwork(observation_spec, global_network,
                                       arm_network)


@gin.configurable
class GlobalAndArmCommonTowerNetwork(network.Network):
  """A network that takes global and arm observations and outputs rewards.

  This network takes the output of the global and per-arm networks, and leads
  them through a common network, that in turn outputs reward estimates.
  """

  def __init__(self,
               observation_spec,
               global_network,
               arm_network,
               common_network,
               name='GlobalAndArmCommonTowerNetwork'):
    """Initializes an instance of `GlobalAndArmCommonTowerNetwork`.

    The network architecture contains networks for both the global and the arm
    features. The outputs of these networks are concatenated and led through a
    third (common) network which in turn outputs reward estimates.

    Args:
      observation_spec: The observation spec for the policy that uses this
        network.
      global_network: The network that takes the global features as input.
      arm_network: The network that takes the arm features as input.
      common_network: The network that takes as input the concatenation of the
        outputs of the global and the arm networks.
      name: The name of this instance of `GlobalAndArmCommonTowerNetwork`.
    """
    super(GlobalAndArmCommonTowerNetwork, self).__init__(
        input_tensor_spec=observation_spec, state_spec=(), name=name)
    self._global_network = global_network
    self._arm_network = arm_network
    self._common_network = common_network

  def call(self, observation, step_type=None, network_state=()):
    """Runs the observation through the network."""

    global_obs = observation[bandit_spec_utils.GLOBAL_FEATURE_KEY]
    arm_obs = observation[bandit_spec_utils.PER_ARM_FEATURE_KEY]
    num_actions = tf.shape(arm_obs)[1]

    global_output, global_state = self._global_network(
        global_obs, step_type=step_type, network_state=network_state)
    global_output = tf.tile(
        tf.expand_dims(global_output, axis=1), [1, num_actions, 1])

    arm_output, arm_state = self._arm_network(
        arm_obs, step_type=step_type, network_state=network_state)

    common_input = tf.concat([global_output, arm_output], axis=-1)

    output, state = self._common_network(common_input,
                                         (global_state, arm_state))
    return tf.squeeze(output, axis=-1), state


@gin.configurable
class GlobalAndArmDotProductNetwork(network.Network):
  """A network that takes global and arm observations and outputs rewards.

  This network calculates the dot product of the output of the global and
  per-arm networks and returns them as reward estimates.
  """

  def __init__(self,
               observation_spec,
               global_network,
               arm_network,
               name='GlobalAndArmDotProductNetwork'):
    """Initializes an instance of `GlobalAndArmDotProductNetwork`.

    The network architecture contains networks for both the global and the arm
    features. The reward estimates will be the dot product of the global and per
    arm outputs.

    Args:
      observation_spec: The observation spec for the policy that uses this
        network.
      global_network: The network that takes the global features as input.
      arm_network: The network that takes the arm features as input.
      name: The name of this instance of `GlobalAndArmDotProductNetwork`.
    """
    super(GlobalAndArmDotProductNetwork, self).__init__(
        input_tensor_spec=observation_spec, state_spec=(), name=name)
    self._global_network = global_network
    self._arm_network = arm_network

  def call(self, observation, step_type=None, network_state=()):
    """Runs the observation through the network."""

    global_obs = observation[bandit_spec_utils.GLOBAL_FEATURE_KEY]
    arm_obs = observation[bandit_spec_utils.PER_ARM_FEATURE_KEY]

    global_output, global_state = self._global_network(
        global_obs, step_type=step_type, network_state=network_state)

    arm_output, arm_state = self._arm_network(
        arm_obs, step_type=step_type, network_state=network_state)

    dot_product = tf.linalg.matvec(arm_output, global_output)
    return dot_product, global_state + arm_state
