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

"""Network that takes as input global and per-arm features, and outputs rewards."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.bandits.specs import utils as bandit_spec_utils
from tf_agents.networks import network
from tf_agents.networks import q_network
from tf_agents.specs import tensor_spec


def create_feed_forward_per_arm_network(observation_spec, global_layers,
                                        arm_layers, common_layers):
  """Creates a reward network that takes global and arm features.

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

  def _create_action_spec(output_dim):
    return tensor_spec.BoundedTensorSpec(
        shape=(), minimum=0, maximum=output_dim - 1, dtype=tf.int32)

  global_output_dim = global_layers[-1]
  global_network = q_network.QNetwork(
      input_tensor_spec=observation_spec[bandit_spec_utils.GLOBAL_FEATURE_KEY],
      action_spec=_create_action_spec(global_output_dim),
      fc_layer_params=global_layers[:0])
  arm_output_dim = arm_layers[-1]
  one_dim_per_arm_obs = tensor_spec.TensorSpec(
      shape=observation_spec[bandit_spec_utils.PER_ARM_FEATURE_KEY].shape[1:],
      dtype=tf.float32)
  arm_network = q_network.QNetwork(
      input_tensor_spec=one_dim_per_arm_obs,
      action_spec=_create_action_spec(arm_output_dim),
      fc_layer_params=arm_layers[:0])
  common_input_dim = global_output_dim + arm_output_dim
  common_input_spec = tensor_spec.TensorSpec(
      shape=(common_input_dim,), dtype=tf.float32)
  common_network = q_network.QNetwork(
      input_tensor_spec=common_input_spec,
      action_spec=_create_action_spec(1),
      fc_layer_params=common_layers)
  return GlobalAndArmFeatureNetwork(observation_spec, global_network,
                                    arm_network, common_network)


@gin.configurable
class GlobalAndArmFeatureNetwork(network.Network):
  """A network that takes global and arm observations and outputs rewards."""

  def __init__(self,
               observation_spec,
               global_network,
               arm_network,
               common_network,
               name='GlobalAndArmFeatureNetwork'):
    """Initializes an instance of `GlobalAndArmFeatureNetwork`.

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
      name: The name of this instance of `GlobalAndArmFeatureNetwork`.
    """
    super(GlobalAndArmFeatureNetwork, self).__init__(
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
