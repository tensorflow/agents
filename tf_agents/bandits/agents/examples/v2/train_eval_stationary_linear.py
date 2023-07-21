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

"""End-to-end test for bandit training under stationary linear environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
from absl import app
from absl import flags

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.bandits.agents import exp3_mixture_agent
from tf_agents.bandits.agents import lin_ucb_agent
from tf_agents.bandits.agents import linear_thompson_sampling_agent as lin_ts_agent
from tf_agents.bandits.agents import neural_boltzmann_agent
from tf_agents.bandits.agents import neural_epsilon_greedy_agent
from tf_agents.bandits.agents.examples.v2 import trainer
from tf_agents.bandits.environments import environment_utilities
from tf_agents.bandits.environments import stationary_stochastic_py_environment as sspe
from tf_agents.bandits.metrics import tf_metrics as tf_bandit_metrics
from tf_agents.environments import tf_py_environment
from tf_agents.environments import wrappers
from tf_agents.networks import q_network
from tf_agents.policies import utils as policy_utilities

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_enum(
    'agent', 'LinUCB',
    ['LinUCB', 'LinTS', 'epsGreedy', 'Mix', 'Boltzmann', 'BoltzmannGumbel'],
    'Which agent to use. Possible values are `LinUCB` and `LinTS`, `epsGreedy`,'
    ' and `Mix`.'
)
flags.DEFINE_bool('normalize_reward_fns', False, 'Whether to normalize the '
                  'reward functions so that rewards are close to being in '
                  '[0, 1].')
flags.DEFINE_integer(
    'num_disabled_actions', 0,
    'If non-zero, there will be extra actions that are always disabled.')

FLAGS = flags.FLAGS

BATCH_SIZE = 8
CONTEXT_DIM = 15
NUM_ACTIONS = 5
REWARD_NOISE_VARIANCE = 0.01
TRAINING_LOOPS = 2000
STEPS_PER_LOOP = 2
AGENT_ALPHA = 0.1
TEMPERATURE = 0.1

EPSILON = 0.05
LAYERS = (50, 50, 50)
LR = 0.001


def main(unused_argv):
  tf.compat.v1.enable_v2_behavior()  # The trainer only runs with V2 enabled.

  with tf.device('/CPU:0'):  # due to b/128333994
    if FLAGS.normalize_reward_fns:
      action_reward_fns = (
          environment_utilities.normalized_sliding_linear_reward_fn_generator(
              CONTEXT_DIM, NUM_ACTIONS, REWARD_NOISE_VARIANCE))
    else:
      action_reward_fns = (
          environment_utilities.sliding_linear_reward_fn_generator(
              CONTEXT_DIM, NUM_ACTIONS, REWARD_NOISE_VARIANCE))

    env = sspe.StationaryStochasticPyEnvironment(
        functools.partial(
            environment_utilities.context_sampling_fn,
            batch_size=BATCH_SIZE,
            context_dim=CONTEXT_DIM),
        action_reward_fns,
        batch_size=BATCH_SIZE)
    mask_split_fn = None
    if FLAGS.num_disabled_actions > 0:
      mask_split_fn = lambda x: (x[0], x[1])
      env = wrappers.ExtraDisabledActionsWrapper(env,
                                                 FLAGS.num_disabled_actions)
    environment = tf_py_environment.TFPyEnvironment(env)

    optimal_reward_fn = functools.partial(
        environment_utilities.tf_compute_optimal_reward,
        per_action_reward_fns=action_reward_fns)

    optimal_action_fn = functools.partial(
        environment_utilities.tf_compute_optimal_action,
        per_action_reward_fns=action_reward_fns)

    network_input_spec = environment.time_step_spec().observation
    if FLAGS.num_disabled_actions > 0:

      def _apply_only_to_observation(fn):
        def result_fn(obs):
          return fn(obs[0])
        return result_fn

      optimal_action_fn = _apply_only_to_observation(optimal_action_fn)
      optimal_reward_fn = _apply_only_to_observation(optimal_reward_fn)
      network_input_spec = network_input_spec[0]

    network = q_network.QNetwork(
        input_tensor_spec=network_input_spec,
        action_spec=environment.action_spec(),
        fc_layer_params=LAYERS)

    if FLAGS.agent == 'LinUCB':
      agent = lin_ucb_agent.LinearUCBAgent(
          time_step_spec=environment.time_step_spec(),
          action_spec=environment.action_spec(),
          alpha=AGENT_ALPHA,
          dtype=tf.float32,
          observation_and_action_constraint_splitter=mask_split_fn)
    elif FLAGS.agent == 'LinTS':
      agent = lin_ts_agent.LinearThompsonSamplingAgent(
          time_step_spec=environment.time_step_spec(),
          action_spec=environment.action_spec(),
          alpha=AGENT_ALPHA,
          dtype=tf.float32,
          observation_and_action_constraint_splitter=mask_split_fn)
    elif FLAGS.agent == 'epsGreedy':
      agent = neural_epsilon_greedy_agent.NeuralEpsilonGreedyAgent(
          time_step_spec=environment.time_step_spec(),
          action_spec=environment.action_spec(),
          reward_network=network,
          optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=LR),
          epsilon=EPSILON,
          observation_and_action_constraint_splitter=mask_split_fn)
    elif FLAGS.agent == 'Boltzmann':
      train_step_counter = tf.compat.v1.train.get_or_create_global_step()
      boundaries = [500]
      temp_values = [1000.0, TEMPERATURE]
      temp_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
          boundaries, temp_values)
      def _temperature_fn():
        # Any variable used in the function needs to be saved in the policy.
        # This is true by default for the `train_step_counter`.
        return temp_schedule(train_step_counter)
      agent = neural_boltzmann_agent.NeuralBoltzmannAgent(
          time_step_spec=environment.time_step_spec(),
          action_spec=environment.action_spec(),
          reward_network=network,
          temperature=_temperature_fn,
          optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=LR),
          observation_and_action_constraint_splitter=mask_split_fn,
          train_step_counter=train_step_counter)
      # This is needed, otherwise the PolicySaver complains.
      agent.policy.step = train_step_counter
    elif FLAGS.agent == 'BoltzmannGumbel':
      num_samples_list = [tf.compat.v2.Variable(
          0, dtype=tf.int64,
          name='num_samples_{}'.format(k)) for k in range(NUM_ACTIONS)]
      agent = neural_boltzmann_agent.NeuralBoltzmannAgent(
          time_step_spec=environment.time_step_spec(),
          action_spec=environment.action_spec(),
          reward_network=network,
          boltzmann_gumbel_exploration_constant=250.0,
          optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=LR),
          observation_and_action_constraint_splitter=mask_split_fn,
          num_samples_list=num_samples_list)
    elif FLAGS.agent == 'Mix':
      assert FLAGS.num_disabled_actions == 0, (
          'Extra actions with mixture agent not supported.')

      emit_policy_info = policy_utilities.InfoFields.PREDICTED_REWARDS_MEAN
      agent_linucb = lin_ucb_agent.LinearUCBAgent(
          time_step_spec=environment.time_step_spec(),
          action_spec=environment.action_spec(),
          emit_policy_info=emit_policy_info,
          alpha=AGENT_ALPHA,
          dtype=tf.float32)
      agent_lints = lin_ts_agent.LinearThompsonSamplingAgent(
          time_step_spec=environment.time_step_spec(),
          action_spec=environment.action_spec(),
          emit_policy_info=emit_policy_info,
          alpha=AGENT_ALPHA,
          dtype=tf.float32)
      agent_epsgreedy = neural_epsilon_greedy_agent.NeuralEpsilonGreedyAgent(
          time_step_spec=environment.time_step_spec(),
          action_spec=environment.action_spec(),
          reward_network=network,
          optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=LR),
          emit_policy_info=emit_policy_info,
          epsilon=EPSILON)
      agent = exp3_mixture_agent.Exp3MixtureAgent(
          (agent_linucb, agent_lints, agent_epsgreedy))

    regret_metric = tf_bandit_metrics.RegretMetric(optimal_reward_fn)
    suboptimal_arms_metric = tf_bandit_metrics.SuboptimalArmsMetric(
        optimal_action_fn)

    trainer.train(
        root_dir=FLAGS.root_dir,
        agent=agent,
        environment=environment,
        training_loops=TRAINING_LOOPS,
        steps_per_loop=STEPS_PER_LOOP,
        additional_metrics=[regret_metric, suboptimal_arms_metric])


if __name__ == '__main__':
  app.run(main)
