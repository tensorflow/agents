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

"""Runs training and eval on CQL-SAC on D4RL using the Actor-Learner API.

All default hyperparameters in train_eval come from the CQL paper:
https://arxiv.org/abs/2006.04779
"""

import os

from typing import Callable, Dict, Optional, Tuple, Union

from absl import app
from absl import flags
from absl import logging

import gin
import numpy as np
import reverb
import rlds
import tensorflow as tf

from tf_agents.agents.cql import cql_sac_agent
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.environments import tf_py_environment
from tf_agents.examples.cql_sac.kumar20.d4rl_utils import load_d4rl
from tf_agents.metrics import py_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.replay_buffers import rlds_to_reverb
from tf_agents.specs import tensor_spec
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils
from tf_agents.trajectories import trajectory
from tf_agents.typing import types

FLAGS = flags.FLAGS

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
_REVERB_PORT = flags.DEFINE_integer(
    'reverb_port', None,
    'Port for reverb server, if None, use a randomly chosen unused port.')
flags.DEFINE_string('env_name', 'antmaze-medium-play-v0',
                    'Name of the environment.')
_DATASET_NAME = flags.DEFINE_string(
    'dataset_name', 'd4rl_antmaze/medium-play-v0',
    'RLDS dataset name. Please select the RLDS dataset'
    'corresponding to D4RL environment chosen for training.')
flags.DEFINE_integer('learner_iterations_per_call', 500,
                     'Iterations per learner run call.')
flags.DEFINE_integer('policy_save_interval', 10000, 'Policy save interval.')
flags.DEFINE_integer('eval_interval', 10000, 'Evaluation interval.')
flags.DEFINE_integer('summary_interval', 1000, 'Summary interval.')
flags.DEFINE_integer('num_gradient_updates', 1000000,
                     'Total number of train iterations to perform.')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding parameters.')
_DATA_TAKE = flags.DEFINE_integer(
    'data_take', None, 'Number of steps to take for training '
    'from RLDS dataset. If not specified, all steps are used '
    'for training.')

_SEQUENCE_LENGTH = 2
_STRIDE_LENGTH = 1


@gin.configurable
def train_eval(
    root_dir: str,
    env_name: str,
    dataset_name: str,
    load_dataset_fn: Optional[Callable[[str], tf.data.Dataset]] = rlds.load,
    # Training params
    tpu: bool = False,
    use_gpu: bool = False,
    num_gradient_updates: int = 1000000,
    actor_fc_layers: Tuple[int, ...] = (256, 256),
    critic_joint_fc_layers: Tuple[int, ...] = (256, 256, 256),
    # Agent params
    batch_size: int = 256,
    bc_steps: int = 0,
    actor_learning_rate: types.Float = 3e-5,
    critic_learning_rate: types.Float = 3e-4,
    alpha_learning_rate: types.Float = 3e-4,
    reward_scale_factor: types.Float = 1.0,
    cql_alpha_learning_rate: types.Float = 3e-4,
    cql_alpha: types.Float = 5.0,
    cql_tau: types.Float = 10.0,
    num_cql_samples: int = 10,
    reward_noise_variance: Union[types.Float, tf.Variable] = 0.0,
    include_critic_entropy_term: bool = False,
    use_lagrange_cql_alpha: bool = True,
    log_cql_alpha_clipping: Optional[Tuple[types.Float, types.Float]] = None,
    softmax_temperature: types.Float = 1.0,
    # Data and Reverb Replay Buffer params
    reward_shift: types.Float = 0.0,
    action_clipping: Optional[Tuple[types.Float, types.Float]] = None,
    data_shuffle_buffer_size: int = 100,
    data_prefetch: int = 10,
    data_take: Optional[int] = None,
    pad_end_of_episodes: bool = False,
    reverb_port: Optional[int] = None,
    min_rate_limiter: int = 1,
    # Others
    policy_save_interval: int = 10000,
    eval_interval: int = 10000,
    summary_interval: int = 1000,
    learner_iterations_per_call: int = 1,
    eval_episodes: int = 10,
    debug_summaries: bool = False,
    summarize_grads_and_vars: bool = False,
    seed: Optional[int] = None) -> None:
  """Trains and evaluates CQL-SAC.

  Args:
    root_dir: Training eval directory
    env_name: Environment to train on.
    dataset_name: RLDS dataset name for the envronment to train.
    load_dataset_fn: A function that will return an instance of a
      tf.data.Dataset for RLDS data to be used for training.
    tpu: Whether to use TPU for training.
    use_gpu: Whether to use GPU for training.
    num_gradient_updates: Number of gradient updates for training.
    actor_fc_layers: Optional list of fully_connected parameters for actor
      distribution network, where each item is the number of units in the layer.
    critic_joint_fc_layers: Optional list of fully connected parameters after
      merging observations and actions in critic, where each item is the number
      of units in the layer.
    batch_size: Batch size for sampling data from Reverb Replay Buffer.
    bc_steps: Number of behavioral cloning steps.
    actor_learning_rate: The learning rate for the actor network. It is used in
      Adam optimiser for actor network.
    critic_learning_rate: The learning rate for the critic network. It is used
      in Adam optimiser for critic network.
    alpha_learning_rate: The learning rate to tune cql alpha. It is used in Adam
      optimiser for cql alpha.
    reward_scale_factor: Multiplicative scale for the reward.
    cql_alpha_learning_rate: The learning rate to tune cql_alpha.
    cql_alpha: The weight on CQL loss. This can be a tf.Variable.
    cql_tau: The threshold for the expected difference in Q-values which
      determines the tuning of cql_alpha.
    num_cql_samples: Number of samples for importance sampling in CQL.
    reward_noise_variance: The noise variance to introduce to the rewards.
    include_critic_entropy_term: Whether to include the entropy term in the
      target for the critic loss.
    use_lagrange_cql_alpha: Whether to use a Lagrange threshold to tune
      cql_alpha during training.
    log_cql_alpha_clipping: (Minimum, maximum) values to clip log CQL alpha.
    softmax_temperature: Temperature value which weights Q-values before the
      `cql_loss` logsumexp calculation.
    reward_shift: shift rewards for each experience sample by the value provided
    action_clipping: Clip actions for each experience sample
    data_shuffle_buffer_size: Shuffle buffer size for the interleaved dataset.
    data_prefetch: Number of data point to prefetch for training from Reverb
      Replay Buffer.
    data_take: Number of steps to take for training from RLDS dataset. If not
      specified, all steps are used for training.
    pad_end_of_episodes: Whether to pad end of episodes.
    reverb_port: Port to start the Reverb server. if not provided, randomly
      chosen port used.
    min_rate_limiter: Reverb min rate limiter.
    policy_save_interval: How often, in train steps, the trigger will save.
    eval_interval: Number of train steps in between evaluations.
    summary_interval: Number of train steps in between summaries.
    learner_iterations_per_call: Iterations per learner run call.
    eval_episodes: Number of episodes evaluated per run call.
    debug_summaries: A bool to gather debug summaries.
    summarize_grads_and_vars: If True, gradient and network variable summaries
      will be written during training.
    seed: Optional seed for tf.random.
  """
  logging.info('Training CQL-SAC on: %s', env_name)
  tf.random.set_seed(seed)
  np.random.seed(seed)
  # Load environment.
  env = load_d4rl(env_name)
  tf_env = tf_py_environment.TFPyEnvironment(env)
  strategy = strategy_utils.get_strategy(tpu, use_gpu)

  # Create dataset of TF-Agents trajectories from RLDS D4RL dataset.
  #
  # The RLDS dataset will be converted to trajectories and pushed to Reverb.
  rlds_data = load_dataset_fn(dataset_name)
  trajectory_data_spec = rlds_to_reverb.create_trajectory_data_spec(rlds_data)
  table_name = 'uniform_table'
  table = reverb.Table(
      name=table_name,
      max_size=data_shuffle_buffer_size,
      sampler=reverb.selectors.Uniform(),
      remover=reverb.selectors.Fifo(),
      rate_limiter=reverb.rate_limiters.MinSize(min_rate_limiter),
      signature=tensor_spec.add_outer_dim(trajectory_data_spec))
  reverb_server = reverb.Server([table], port=reverb_port)
  reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
      trajectory_data_spec,
      sequence_length=_SEQUENCE_LENGTH,
      table_name=table_name,
      local_server=reverb_server)
  rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
      reverb_replay.py_client,
      table_name,
      sequence_length=_SEQUENCE_LENGTH,
      stride_length=_STRIDE_LENGTH,
      pad_end_of_episodes=pad_end_of_episodes)

  def _transform_episode(episode: tf.data.Dataset) -> tf.data.Dataset:
    """Apply reward_shift and action_clipping to RLDS episode.

    Args:
      episode: An RLDS episode dataset of RLDS steps datasets.

    Returns:
      An RLDS episode after applying action clipping and reward shift.
    """

    def _transform_step(
        rlds_step: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
      """Apply reward_shift and action_clipping to RLDS step.

      Args:
        rlds_step: An RLDS step is a dictionary of tensors containing is_first,
          is_last, observation, action, reward, is_terminal, and discount.

      Returns:
        An RLDS step after applying action clipping and reward shift.
      """
      rlds_step[rlds.REWARD] = rlds_step[rlds.REWARD] + reward_shift
      if action_clipping:
        rlds_step[rlds.ACTION] = tf.clip_by_value(
            rlds_step[rlds.ACTION],
            clip_value_min=action_clipping[0],
            clip_value_max=action_clipping[1])
      return rlds_step

    episode[rlds.STEPS] = episode[rlds.STEPS].map(_transform_step)
    return episode

  if data_take:
    rlds_data = rlds_data.take(data_take)

  if reward_shift or action_clipping:
    rlds_data = rlds_data.map(_transform_episode)

  rlds_to_reverb.push_rlds_to_reverb(rlds_data, rb_observer)

  def _experience_dataset() -> tf.data.Dataset:
    """Reads and returns the experiences dataset from Reverb Replay Buffer."""
    return reverb_replay.as_dataset(
        sample_batch_size=batch_size,
        num_steps=_SEQUENCE_LENGTH).prefetch(data_prefetch)

  # Create agent.
  time_step_spec = tf_env.time_step_spec()
  observation_spec = time_step_spec.observation
  action_spec = tf_env.action_spec()
  with strategy.scope():
    train_step = train_utils.create_train_step()

    actor_net = actor_distribution_network.ActorDistributionNetwork(
        observation_spec,
        action_spec,
        fc_layer_params=actor_fc_layers,
        continuous_projection_net=tanh_normal_projection_network
        .TanhNormalProjectionNetwork)

    critic_net = critic_network.CriticNetwork(
        (observation_spec, action_spec),
        joint_fc_layer_params=critic_joint_fc_layers,
        kernel_initializer='glorot_uniform',
        last_kernel_initializer='glorot_uniform')

    agent = cql_sac_agent.CqlSacAgent(
        time_step_spec,
        action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=tf.keras.optimizers.Adam(
            learning_rate=actor_learning_rate),
        critic_optimizer=tf.keras.optimizers.Adam(
            learning_rate=critic_learning_rate),
        alpha_optimizer=tf.keras.optimizers.Adam(
            learning_rate=alpha_learning_rate),
        cql_alpha=cql_alpha,
        num_cql_samples=num_cql_samples,
        include_critic_entropy_term=include_critic_entropy_term,
        use_lagrange_cql_alpha=use_lagrange_cql_alpha,
        cql_alpha_learning_rate=cql_alpha_learning_rate,
        target_update_tau=5e-3,
        target_update_period=1,
        random_seed=seed,
        cql_tau=cql_tau,
        reward_noise_variance=reward_noise_variance,
        num_bc_steps=bc_steps,
        td_errors_loss_fn=tf.math.squared_difference,
        gamma=0.99,
        reward_scale_factor=reward_scale_factor,
        gradient_clipping=None,
        log_cql_alpha_clipping=log_cql_alpha_clipping,
        softmax_temperature=softmax_temperature,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=train_step)
    agent.initialize()

  # Create learner.
  saved_model_dir = os.path.join(root_dir, learner.POLICY_SAVED_MODEL_DIR)
  collect_env_step_metric = py_metrics.EnvironmentSteps()
  learning_triggers = [
      triggers.PolicySavedModelTrigger(
          saved_model_dir,
          agent,
          train_step,
          interval=policy_save_interval,
          metadata_metrics={
              triggers.ENV_STEP_METADATA_KEY: collect_env_step_metric
          }),
      triggers.StepPerSecondLogTrigger(train_step, interval=100)
  ]
  cql_learner = learner.Learner(
      root_dir,
      train_step,
      agent,
      experience_dataset_fn=_experience_dataset,
      triggers=learning_triggers,
      summary_interval=summary_interval,
      strategy=strategy)

  # Create actor for evaluation.
  tf_greedy_policy = greedy_policy.GreedyPolicy(agent.policy)
  eval_greedy_policy = py_tf_eager_policy.PyTFEagerPolicy(
      tf_greedy_policy, use_tf_function=True)
  eval_actor = actor.Actor(
      env,
      eval_greedy_policy,
      train_step,
      metrics=actor.eval_metrics(eval_episodes),
      summary_dir=os.path.join(root_dir, 'eval'),
      episodes_per_run=eval_episodes)

  # Run.
  dummy_trajectory = trajectory.mid((), (), (), 0., 1.)
  num_learner_iterations = int(num_gradient_updates /
                               learner_iterations_per_call)
  for _ in range(num_learner_iterations):
    # Mimic collecting environment steps since we loaded a static dataset.
    for _ in range(learner_iterations_per_call):
      collect_env_step_metric(dummy_trajectory)

    cql_learner.run(iterations=learner_iterations_per_call)
    if eval_interval and train_step.numpy() % eval_interval == 0:
      eval_actor.run_and_log()


def main(_):
  logging.set_verbosity(logging.INFO)

  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)

  train_eval(
      root_dir=FLAGS.root_dir,
      dataset_name=_DATASET_NAME.value,
      env_name=FLAGS.env_name,
      tpu=FLAGS.tpu,
      use_gpu=FLAGS.use_gpu,
      num_gradient_updates=FLAGS.num_gradient_updates,
      policy_save_interval=FLAGS.policy_save_interval,
      eval_interval=FLAGS.eval_interval,
      summary_interval=FLAGS.summary_interval,
      learner_iterations_per_call=FLAGS.learner_iterations_per_call,
      reverb_port=_REVERB_PORT.value,
      data_take=_DATA_TAKE.value)


if __name__ == '__main__':
  app.run(main)
