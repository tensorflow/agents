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

"""Unit tests for the Actor classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import portpicker
from six.moves import range
import tensorflow.compat.v2 as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import suite_gym
from tf_agents.metrics import py_metrics
from tf_agents.policies import py_tf_eager_policy
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.train import actor
from tf_agents.train.utils import replay_buffer_utils
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import test_utils as dist_test_utils
from tf_agents.train.utils import train_utils
from tf_agents.utils import test_utils


class ActorTest(test_utils.TestCase):
  """A set of tests for the Actor classes."""

  def _build_components(self, rb_port):
    env = suite_gym.load('CartPole-v0')

    _, action_tensor_spec, time_step_tensor_spec = (
        spec_utils.get_tensor_specs(env))
    train_step = train_utils.create_train_step()

    q_net = dist_test_utils.build_dummy_sequential_net(
        fc_layer_params=(100,), action_spec=action_tensor_spec)

    agent = dqn_agent.DqnAgent(
        time_step_tensor_spec,
        action_tensor_spec,
        q_network=q_net,
        optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.001),
        train_step_counter=train_step)

    replay_buffer, rb_observer = (
        replay_buffer_utils.get_reverb_buffer_and_observer(
            agent.collect_data_spec,
            sequence_length=2,
            replay_capacity=1000,
            port=rb_port))

    return env, agent, train_step, replay_buffer, rb_observer

  def testActorRun(self):
    rb_port = portpicker.pick_unused_port(portserver_address='localhost')

    env, agent, train_step, replay_buffer, rb_observer = (
        self._build_components(rb_port))

    tf_collect_policy = agent.collect_policy
    collect_policy = py_tf_eager_policy.PyTFEagerPolicy(tf_collect_policy,
                                                        use_tf_function=True)
    test_actor = actor.Actor(
        env,
        collect_policy,
        train_step,
        steps_per_run=1,
        observers=[rb_observer])

    self.assertEqual(replay_buffer.num_frames(), 0)
    for _ in range(10):
      test_actor.run()
    self.assertGreater(replay_buffer.num_frames(), 0)

  def testRefereneMetricsNotInObservers(self):
    rb_port = portpicker.pick_unused_port(portserver_address='localhost')

    env, agent, train_step, _, rb_observer = (
        self._build_components(rb_port))

    temp_dir = self.create_tempdir().full_path
    tf_collect_policy = agent.collect_policy
    collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
        tf_collect_policy, use_tf_function=True)
    metrics = metrics = actor.collect_metrics(buffer_size=1)
    step_metric = py_metrics.EnvironmentSteps()
    test_actor = actor.Actor(
        env,
        collect_policy,
        train_step,
        steps_per_run=1,
        metrics=metrics,
        reference_metrics=[step_metric],
        summary_dir=temp_dir,
        observers=[rb_observer])

    self.assertNotIn(step_metric, test_actor._observers)

  def testCollectLocalPyActorRun(self):
    rb_port = portpicker.pick_unused_port(portserver_address='localhost')

    env, agent, train_step, replay_buffer, rb_observer = (
        self._build_components(rb_port))

    temp_dir = self.create_tempdir().full_path
    tf_collect_policy = agent.collect_policy
    collect_policy = py_tf_eager_policy.PyTFEagerPolicy(tf_collect_policy,
                                                        use_tf_function=True)
    test_actor = actor.Actor(
        env,
        collect_policy,
        train_step,
        steps_per_run=1,
        metrics=actor.collect_metrics(buffer_size=1),
        summary_dir=temp_dir,
        observers=[rb_observer])

    self.assertEqual(replay_buffer.num_frames(), 0)
    for _ in range(10):
      test_actor.run()
    self.assertGreater(replay_buffer.num_frames(), 0)

  def testEvalLocalPyActorRun(self):
    rb_port = portpicker.pick_unused_port(portserver_address='localhost')

    env, agent, train_step, replay_buffer, _ = (
        self._build_components(rb_port))

    temp_dir = self.create_tempdir().full_path
    tf_collect_policy = agent.collect_policy
    collect_policy = py_tf_eager_policy.PyTFEagerPolicy(tf_collect_policy,
                                                        use_tf_function=True)
    test_actor = actor.Actor(
        env,
        collect_policy,
        train_step,
        episodes_per_run=1,
        metrics=actor.eval_metrics(buffer_size=1),
        summary_dir=temp_dir,
    )

    self.assertEqual(replay_buffer.num_frames(), 0)
    for _ in range(2):
      test_actor.run()
    self.assertEqual(replay_buffer.num_frames(), 0)
    self.assertGreater(test_actor._metrics[0].result(), 0)

if __name__ == '__main__':
  multiprocessing.handle_test_main(tf.test.main)
