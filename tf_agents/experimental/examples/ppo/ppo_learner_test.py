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

"""Tests for tf_agents.google.experimental.examples.ppo.ppo_learner.

Verifies that the expected training data is passed into the agent from the
PPO learner in different settings.
"""

from absl import flags
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tf_agents.agents import tf_agent
from tf_agents.agents.ppo import ppo_agent
from tf_agents.experimental.examples.ppo import ppo_learner
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import test_utils

FLAGS = flags.FLAGS


# We could not use a mock directly because the Learner requires the agent to
# be `Trackable`.
class FakePPOAgent(ppo_agent.PPOAgent):

  def __init__(self):

    actor_net = actor_distribution_network.ActorDistributionNetwork(
        tensor_spec.TensorSpec(shape=[], dtype=tf.float32),
        tensor_spec.BoundedTensorSpec([1], tf.float32, -1, 1),
        fc_layer_params=(1,),
        activation_fn=tf.nn.tanh)
    value_net = value_network.ValueNetwork(
        tensor_spec.TensorSpec(shape=[], dtype=tf.float32),
        fc_layer_params=(1,))

    super(FakePPOAgent, self).__init__(
        time_step_spec=ts.time_step_spec(
            tensor_spec.TensorSpec(shape=[], dtype=tf.float32)),
        action_spec=tensor_spec.BoundedTensorSpec([1], tf.float32, -1, 1),
        actor_net=actor_net,
        value_net=value_net,
        # Ensures value_prediction, return and advantage are included as parts
        # of the training_data_spec.
        compute_value_and_advantage_in_train=False,
        update_normalizers_in_train=False,
    )
    # There is an artifical call on `_train` during the initialization which
    # ensures that the variables of the optimizer are initialized. This is
    # excluded from the call count.
    self.train_called_times = -1
    self.experiences = []

  def _train(self, experience, weights):
    self.train_called_times += 1
    # The first call is an artificial one and it corresponds to the optimizer
    # variable initialization, and it does not run on a true experience, so it
    # is excluded.
    if self.train_called_times > 0:
      self.experiences.append(experience)
    return tf_agent.LossInfo(0., 0.)


def _create_trajectories(n_time_steps, batch_size):
  # Observation looks like:
  # [[ 0.,  1., ... n_time_steps.],
  #  [10., 11., ... n_time_steps.],
  #  [20., 21., ... n_time_steps.],
  #  [ ...                       ],
  #  [10*batch_size., ... 10*batch_size+n_time_steps.]]
  observation_array = np.asarray([
      np.arange(n_time_steps) + 10 * i for i in range(batch_size)
  ])
  observations = tf.convert_to_tensor(observation_array, dtype=tf.float32)

  default_tensor = tf.constant(
      [[1] * n_time_steps] * batch_size, dtype=tf.float32)
  mid_time_step_val = ts.StepType.MID.tolist()
  time_steps = ts.TimeStep(
      step_type=tf.constant(
          [[mid_time_step_val] * n_time_steps] * batch_size, dtype=tf.int32),
      reward=default_tensor,
      discount=default_tensor,
      observation=observations)
  actions = tf.constant([[[1]] * n_time_steps] * batch_size, dtype=tf.float32)
  policy_info = {
      'dist_params': {
          'loc':
              tf.constant(
                  [[[1]] * n_time_steps] * batch_size, dtype=tf.float32),
          'scale':
              tf.constant(
                  [[[1]] * n_time_steps] * batch_size, dtype=tf.float32)
      },
      'value_prediction': default_tensor,
      'return': default_tensor,
      'advantage': default_tensor,
  }
  return trajectory.Trajectory(time_steps.step_type, observations, actions,
                               policy_info, time_steps.step_type,
                               time_steps.reward, time_steps.discount)


def _concat_and_flatten(traj, multiplier):
  """Concatenate tensors in the input trajectory by `multiplier` times.

  Args:
    traj: a `Trajectory` shaped [batch_size, num_steps, ...].
    multiplier: the number of times to concatenate the input trajectory.
  Returns:
    a flattened `Trajectory` shaped [multiplier * batch_size * num_steps, ...].
  """
  def concat_and_flatten_tensor(tensor):
    multipled_component_list = [tensor] * multiplier
    concat_tensor = tf.concat(multipled_component_list, axis=0)

    first_dim = multiplier * tensor.shape[0] * tensor.shape[1]
    other_dims = tensor.shape[2:]
    return tf.reshape(concat_tensor, shape=(first_dim,) + other_dims)

  concated_traj = tf.nest.map_structure(concat_and_flatten_tensor, traj)
  return concated_traj


def _get_expected_minibatch(all_traj, minibatch_size, current_iteration):
  """Get the `Trajectory` containing the expected minibatch.

  Args:
    all_traj: a flattened `Trajectory` without the batch and time dimension.
    minibatch_size: the number of steps included in each minibatch.
    current_iteration: the indx of the current training iteration.

  Returns:
    The expected `Trajectory` shaped [minibatch_size, 1, ...].
  """
  expected_traj = tf.nest.map_structure(
      # pylint: disable=g-long-lambda
      lambda x: x[minibatch_size * current_iteration:minibatch_size *
                  (current_iteration + 1)], all_traj)
  # Add time dimension to be consistent with the input to agent.train.
  expected_traj = tf.nest.map_structure(lambda x: tf.expand_dims(x, 1),
                                        expected_traj)
  return expected_traj


class PpoLearnerTest(parameterized.TestCase, test_utils.TestCase):

  def setUp(self):
    super(PpoLearnerTest, self).setUp()
    # Run in full eager mode in order to inspect the content of the trajectory
    # fed into the fake agent.
    tf.config.experimental_run_functions_eagerly(True)

  def tearDown(self):
    tf.config.experimental_run_functions_eagerly(False)
    super(PpoLearnerTest, self).tearDown()

  @parameterized.named_parameters(
      ('OneEpochMinibatch', 1, 1, 10, 10),
      ('TwoEpochsMinibatch', 2, 1, 10, 20),
      ('ParallelTwoEpochsMinibatch', 2, 3, 10, 60),
      ('OneEpochNoMinibatch', 1, 1, None, 1),
      ('TwoEpochsNoMinibatch', 2, 1, None, 2),
      ('ParallelTwoEpochsNoMinibatch', 2, 3, None, 2),
  )
  def test_one_element_dataset(self, num_epochs, num_parallel_environments,
                               minibatch_size, expected_train_times):
    # Create a dataset with one element that is a length 100 sequence. This
    # simulates a Reverb dataset if only one sequence was collected.
    traj = _create_trajectories(
        n_time_steps=100, batch_size=num_parallel_environments)
    info = ()

    dataset_fn = lambda: tf.data.Dataset.from_tensors((traj, info),)

    fake_agent = FakePPOAgent()

    learner = ppo_learner.PPOLearner(
        root_dir=FLAGS.test_tmpdir,
        train_step=tf.Variable(0, dtype=tf.int32),
        agent=fake_agent,
        experience_dataset_fn=dataset_fn,
        normalization_dataset_fn=dataset_fn,
        num_batches=1,
        num_epochs=num_epochs,
        minibatch_size=minibatch_size,
        # Disable shuffling to have deterministic input into agent.train.
        shuffle_buffer_size=1,
        triggers=None)
    learner.run()

    # Check that fake agent was called the expected number of times.
    self.assertEqual(fake_agent.train_called_times, expected_train_times)

    # Check that agent.train() is receiving the expected trajectories.
    if minibatch_size:
      concated_traj = _concat_and_flatten(traj, multiplier=num_epochs)
      for i in range(expected_train_times):
        expected_traj = _get_expected_minibatch(
            concated_traj, minibatch_size, current_iteration=i)
        received_traj = fake_agent.experiences[i]
        tf.nest.map_structure(self.assertAllClose, received_traj, expected_traj)
    else:
      for i in range(num_epochs):
        expected_traj = traj
        received_traj = fake_agent.experiences[i]
        tf.nest.map_structure(self.assertAllClose, received_traj, expected_traj)

  @parameterized.named_parameters(
      ('OneEpochMinibatch', 1, 1, 10, 12),
      ('TwoEpochsMinibatch', 2, 1, 10, 24),
      ('ParallelTwoEpochsMinibatch', 2, 3, 10, 72),
      ('OneEpochNoMinibatch', 1, 1, None, 3),
      ('TwoEpochsNoMinibatch', 2, 1, None, 6),
      ('ParallelTwoEpochsNoMinibatch', 2, 3, None, 6),
  )
  def test_multi_element_dataset_minibatch(self, num_epochs,
                                           num_parallel_environments,
                                           minibatch_size,
                                           expected_train_times):
    num_episodes = 3
    # Create a dataset with three elements. Each element represents an collected
    # episode of length 40.
    get_shape = lambda x: x.shape
    get_dtype = lambda x: tf.as_dtype(x.dtype)
    traj = _create_trajectories(
        n_time_steps=40, batch_size=num_parallel_environments)
    unused_info = ()
    shapes = tf.nest.map_structure(get_shape, (traj, unused_info))
    dtypes = tf.nest.map_structure(get_dtype, (traj, unused_info))

    def generate_data():
      for _ in range(num_episodes):
        yield (traj, unused_info)

    def dataset_fn():
      return tf.data.Dataset.from_generator(
          generate_data,
          dtypes,
          output_shapes=shapes,
      )

    fake_agent = FakePPOAgent()

    learner = ppo_learner.PPOLearner(
        root_dir=FLAGS.test_tmpdir,
        train_step=tf.Variable(0, dtype=tf.int32),
        agent=fake_agent,
        experience_dataset_fn=dataset_fn,
        normalization_dataset_fn=dataset_fn,
        num_batches=num_episodes,
        num_epochs=num_epochs,
        minibatch_size=minibatch_size,
        # Disable shuffling to have deterministic input into agent.train.
        shuffle_buffer_size=1,
        triggers=None)
    learner.run()

    # Check that fake agent was called the expected number of times.
    self.assertEqual(fake_agent.train_called_times, expected_train_times)

    # Check that agent.train() is receiving the expected trajectories.
    if minibatch_size:
      concated_traj = _concat_and_flatten(
          traj, multiplier=num_episodes * num_epochs)

      for i in range(expected_train_times):
        expected_traj = _get_expected_minibatch(
            concated_traj, minibatch_size, current_iteration=i)
        received_traj = fake_agent.experiences[i]
        tf.nest.map_structure(self.assertAllClose, received_traj, expected_traj)
    else:
      for i in range(expected_train_times):
        expected_traj = traj
        received_traj = fake_agent.experiences[i]
        tf.nest.map_structure(self.assertAllClose, received_traj, expected_traj)


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
