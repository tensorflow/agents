# coding=utf-8
# Copyright 2018 The TFAgents Authors.
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

"""Tests for tf_agents.agents.ddpg.rnn_networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import numpy as np
import tensorflow as tf

from tf_agents.agents.ddpg import rnn_networks
from tf_agents.environments import time_step as ts
from tf_agents.specs import tensor_spec

nest = tf.contrib.framework.nest


class RnnNetworksTest(tf.test.TestCase):

  def test_q_network_lstm_builds(self):
    time_steps = ts.restart(
        tf.constant(0, shape=(5, 10), dtype=tf.float32), batch_size=5)

    policy_state_spec = rnn_networks.get_state_spec()

    def add_batch_dim(spec):
      return tf.constant(
          0, shape=[
              5,
          ] + spec.shape.as_list(), dtype=spec.dtype)

    policy_state = nest.map_structure(add_batch_dim, policy_state_spec)

    time_step_spec = ts.time_step_spec(
        tensor_spec.TensorSpec((10,), dtype=tf.float32))

    action_spec = tensor_spec.BoundedTensorSpec([1], tf.float32, 2., 3.)

    actions, actor_policy_state = rnn_networks.actor_network(
        time_steps,
        action_spec,
        time_step_spec,
        policy_state=policy_state,
        fc_layers=[20, 4]
    )

    q_values, critic_policy_state = rnn_networks.critic_network(
        time_steps,
        actions,
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        policy_state=policy_state,
        observation_fc_layers=[20, 4],
        joint_fc_layers=[20]
    )

    self.assertAllEqual(q_values.shape.as_list(), [5])
    self.assertAllEqual(actions.shape.as_list(),
                        [5] + action_spec.shape.as_list())

    self.assertEqual(2, len(critic_policy_state))
    self.assertAllEqual([(5, 40), (5, 40)],
                        [s.shape for s in critic_policy_state])

    self.assertEqual(2, len(actor_policy_state))
    self.assertAllEqual([(5, 40), (5, 40)],
                        [s.shape for s in actor_policy_state])


if __name__ == '__main__':
  tf.test.main()
