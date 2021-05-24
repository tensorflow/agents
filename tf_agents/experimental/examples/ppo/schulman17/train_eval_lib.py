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

# Lint as: python3
r"""Train/eval PPOClipAgent in Mujoco environments with (Schulman, 17) methods.

To reproduce (Schulman, 17), here we collect fixed length sequences of
`collect_sequence_length` to store in the replay buffer and perform advantage
calculation on. Each sequence can span multiple episodes, separated by a
boundary step. Bootstraping occurs during advantage calculation for the partial
episodes that are part of the sequences.

Note that this isn't necessary. Alternatively, you could instead collect a
specified number of episodes in each training iteration. Each `item` stored in
Reverb tables is a full episode, and advantage calculation happens on full
episodes. As a result, no bootstrapping is required.

All hyperparameters come from the PPO paper
https://arxiv.org/abs/1707.06347.pdf
"""
import os

from absl import logging

import gin
import reverb
import tensorflow.compat.v2 as tf

from tf_agents.agents.ppo import ppo_actor_network
from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.environments import suite_mujoco
from tf_agents.metrics import py_metrics
from tf_agents.networks import value_network
from tf_agents.policies import py_tf_eager_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import ppo_learner
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import train_utils


class ReverbFixedLengthSequenceObserver(
    reverb_utils.ReverbAddTrajectoryObserver):
  """Reverb fixed length sequence observer.

  This is a specialized observer similar to ReverbAddTrajectoryObserver but each
  sequence contains a fixed number of steps and can span multiple episodes. This
  implementation is consistent with (Schulman, 17).

  **Note**: Counting of steps in drivers does not include boundary steps. To
  guarantee only 1 item is pushed to the replay when collecting n steps with a
  `sequence_length` of n make sure to set the `stride_length`.
  """

  def __call__(self, trajectory):
    """Writes the trajectory into the underlying replay buffer.

    Allows trajectory to be a flattened trajectory. No batch dimension allowed.

    Args:
      trajectory: The trajectory to be written which could be (possibly nested)
        trajectory object or a flattened version of a trajectory. It assumes
        there is *no* batch dimension.
    """
    self._writer.append(trajectory)
    self._cached_steps += 1

    self._write_cached_steps()


@gin.configurable
def train_eval(
    root_dir,
    env_name='HalfCheetah-v2',
    # Training params
    num_iterations=1600,
    actor_fc_layers=(64, 64),
    value_fc_layers=(64, 64),
    learning_rate=3e-4,
    collect_sequence_length=2048,
    minibatch_size=64,
    num_epochs=10,
    # Agent params
    importance_ratio_clipping=0.2,
    lambda_value=0.95,
    discount_factor=0.99,
    entropy_regularization=0.,
    value_pred_loss_coef=0.5,
    use_gae=True,
    use_td_lambda_return=True,
    gradient_clipping=0.5,
    value_clipping=None,
    # Replay params
    reverb_port=None,
    replay_capacity=10000,
    # Others
    policy_save_interval=5000,
    summary_interval=1000,
    eval_interval=10000,
    eval_episodes=100,
    debug_summaries=False,
    summarize_grads_and_vars=False):
  """Trains and evaluates PPO (Importance Ratio Clipping).

  Args:
    root_dir: Main directory path where checkpoints, saved_models, and summaries
      will be written to.
    env_name: Name for the Mujoco environment to load.
    num_iterations: The number of iterations to perform collection and training.
    actor_fc_layers: List of fully_connected parameters for the actor network,
      where each item is the number of units in the layer.
    value_fc_layers: : List of fully_connected parameters for the value network,
      where each item is the number of units in the layer.
    learning_rate: Learning rate used on the Adam optimizer.
    collect_sequence_length: Number of steps to take in each collect run.
    minibatch_size: Number of elements in each mini batch. If `None`, the entire
      collected sequence will be treated as one batch.
    num_epochs: Number of iterations to repeat over all collected data per data
      collection step. (Schulman,2017) sets this to 10 for Mujoco, 15 for
      Roboschool and 3 for Atari.
    importance_ratio_clipping: Epsilon in clipped, surrogate PPO objective. For
      more detail, see explanation at the top of the doc.
    lambda_value: Lambda parameter for TD-lambda computation.
    discount_factor: Discount factor for return computation. Default to `0.99`
      which is the value used for all environments from (Schulman, 2017).
    entropy_regularization: Coefficient for entropy regularization loss term.
      Default to `0.0` because no entropy bonus was used in (Schulman, 2017).
    value_pred_loss_coef: Multiplier for value prediction loss to balance with
      policy gradient loss. Default to `0.5`, which was used for all
      environments in the OpenAI baseline implementation. This parameters is
      irrelevant unless you are sharing part of actor_net and value_net. In that
      case, you would want to tune this coeeficient, whose value depends on the
      network architecture of your choice.
    use_gae: If True (default False), uses generalized advantage estimation for
      computing per-timestep advantage. Else, just subtracts value predictions
      from empirical return.
    use_td_lambda_return: If True (default False), uses td_lambda_return for
      training value function; here: `td_lambda_return = gae_advantage +
        value_predictions`. `use_gae` must be set to `True` as well to enable TD
        -lambda returns. If `use_td_lambda_return` is set to True while
        `use_gae` is False, the empirical return will be used and a warning will
        be logged.
    gradient_clipping: Norm length to clip gradients.
    value_clipping: Difference between new and old value predictions are clipped
      to this threshold. Value clipping could be helpful when training
      very deep networks. Default: no clipping.
    reverb_port: Port for reverb server, if None, use a randomly chosen unused
      port.
    replay_capacity: The maximum number of elements for the replay buffer. Items
      will be wasted if this is smalled than collect_sequence_length.
    policy_save_interval: How often, in train_steps, the policy will be saved.
    summary_interval: How often to write data into Tensorboard.
    eval_interval: How often to run evaluation, in train_steps.
    eval_episodes: Number of episodes to evaluate over.
    debug_summaries: Boolean for whether to gather debug summaries.
    summarize_grads_and_vars: If true, gradient summaries will be written.
  """
  collect_env = suite_mujoco.load(env_name)
  eval_env = suite_mujoco.load(env_name)
  num_environments = 1

  observation_tensor_spec, action_tensor_spec, time_step_tensor_spec = (
      spec_utils.get_tensor_specs(collect_env))

  train_step = train_utils.create_train_step()

  actor_net_builder = ppo_actor_network.PPOActorNetwork()
  actor_net = actor_net_builder.create_sequential_actor_net(
      actor_fc_layers, action_tensor_spec)
  value_net = value_network.ValueNetwork(
      observation_tensor_spec,
      fc_layer_params=value_fc_layers,
      kernel_initializer=tf.keras.initializers.Orthogonal())

  current_iteration = tf.Variable(0, dtype=tf.int64)
  def learning_rate_fn():
    # Linearly decay the learning rate.
    return learning_rate * (1 - current_iteration / num_iterations)

  agent = ppo_clip_agent.PPOClipAgent(
      time_step_tensor_spec,
      action_tensor_spec,
      optimizer=tf.compat.v1.train.AdamOptimizer(
          learning_rate=learning_rate_fn, epsilon=1e-5),
      actor_net=actor_net,
      value_net=value_net,
      importance_ratio_clipping=importance_ratio_clipping,
      lambda_value=lambda_value,
      discount_factor=discount_factor,
      entropy_regularization=entropy_regularization,
      value_pred_loss_coef=value_pred_loss_coef,
      # This is a legacy argument for the number of times we repeat the data
      # inside of the train function, incompatible with mini batch learning.
      # We set the epoch number from the replay buffer and tf.Data instead.
      num_epochs=1,
      use_gae=use_gae,
      use_td_lambda_return=use_td_lambda_return,
      gradient_clipping=gradient_clipping,
      value_clipping=value_clipping,
      # TODO(b/150244758): Default compute_value_and_advantage_in_train to False
      # after Reverb open source.
      compute_value_and_advantage_in_train=False,
      # Skips updating normalizers in the agent, as it's handled in the learner.
      update_normalizers_in_train=False,
      debug_summaries=debug_summaries,
      summarize_grads_and_vars=summarize_grads_and_vars,
      train_step_counter=train_step)
  agent.initialize()

  reverb_server = reverb.Server(
      [
          reverb.Table(  # Replay buffer storing experience for training.
              name='training_table',
              sampler=reverb.selectors.Fifo(),
              remover=reverb.selectors.Fifo(),
              rate_limiter=reverb.rate_limiters.MinSize(1),
              max_size=replay_capacity,
              max_times_sampled=1,
          ),
          reverb.Table(  # Replay buffer storing experience for normalization.
              name='normalization_table',
              sampler=reverb.selectors.Fifo(),
              remover=reverb.selectors.Fifo(),
              rate_limiter=reverb.rate_limiters.MinSize(1),
              max_size=replay_capacity,
              max_times_sampled=1,
          )
      ],
      port=reverb_port)

  # Create the replay buffer.
  reverb_replay_train = reverb_replay_buffer.ReverbReplayBuffer(
      agent.collect_data_spec,
      sequence_length=collect_sequence_length,
      table_name='training_table',
      server_address='localhost:{}'.format(reverb_server.port),
      # The only collected sequence is used to populate the batches.
      max_cycle_length=1,
      rate_limiter_timeout_ms=1000)
  reverb_replay_normalization = reverb_replay_buffer.ReverbReplayBuffer(
      agent.collect_data_spec,
      sequence_length=collect_sequence_length,
      table_name='normalization_table',
      server_address='localhost:{}'.format(reverb_server.port),
      # The only collected sequence is used to populate the batches.
      max_cycle_length=1,
      rate_limiter_timeout_ms=1000)

  rb_observer = ReverbFixedLengthSequenceObserver(
      reverb_replay_train.py_client, ['training_table', 'normalization_table'],
      sequence_length=collect_sequence_length,
      stride_length=collect_sequence_length)

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
      triggers.StepPerSecondLogTrigger(train_step, interval=summary_interval),
  ]

  def training_dataset_fn():
    return reverb_replay_train.as_dataset(
        sample_batch_size=num_environments,
        sequence_preprocess_fn=agent.preprocess_sequence)

  def normalization_dataset_fn():
    return reverb_replay_normalization.as_dataset(
        sample_batch_size=num_environments,
        sequence_preprocess_fn=agent.preprocess_sequence)

  agent_learner = ppo_learner.PPOLearner(
      root_dir,
      train_step,
      agent,
      experience_dataset_fn=training_dataset_fn,
      normalization_dataset_fn=normalization_dataset_fn,
      num_samples=1,
      num_epochs=num_epochs,
      minibatch_size=minibatch_size,
      shuffle_buffer_size=collect_sequence_length,
      triggers=learning_triggers)

  tf_collect_policy = agent.collect_policy
  collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
      tf_collect_policy, use_tf_function=True)

  collect_actor = actor.Actor(
      collect_env,
      collect_policy,
      train_step,
      steps_per_run=collect_sequence_length,
      observers=[rb_observer],
      metrics=actor.collect_metrics(buffer_size=10) + [collect_env_step_metric],
      reference_metrics=[collect_env_step_metric],
      summary_dir=os.path.join(root_dir, learner.TRAIN_DIR),
      summary_interval=summary_interval)

  eval_greedy_policy = py_tf_eager_policy.PyTFEagerPolicy(
      agent.policy, use_tf_function=True)

  if eval_interval:
    logging.info('Intial evaluation.')
    eval_actor = actor.Actor(
        eval_env,
        eval_greedy_policy,
        train_step,
        metrics=actor.eval_metrics(eval_episodes),
        reference_metrics=[collect_env_step_metric],
        summary_dir=os.path.join(root_dir, 'eval'),
        episodes_per_run=eval_episodes)

    eval_actor.run_and_log()

  logging.info('Training on %s', env_name)
  last_eval_step = 0
  for i in range(num_iterations):
    collect_actor.run()
    # TODO(b/159615593): Update to use observer.flush.
    # Reset the reverb observer to make sure the data collected is flushed and
    # written to the RB.
    # At this point, there a small number of steps left in the cache because the
    # actor does not count a boundary step as a step, whereas it still gets
    # added to Reverb for training. We throw away those extra steps without
    # padding to align with the paper implementation which never collects them
    # in the first place.
    rb_observer.reset(write_cached_steps=False)
    agent_learner.run()
    reverb_replay_train.clear()
    reverb_replay_normalization.clear()
    current_iteration.assign_add(1)

    # Eval only if `eval_interval` has been set. Then, eval if the current train
    # step is equal or greater than the `last_eval_step` + `eval_interval` or if
    # this is the last iteration. This logic exists because agent_learner.run()
    # does not return after every train step.
    if (eval_interval and
        (agent_learner.train_step_numpy >= eval_interval + last_eval_step
         or i == num_iterations - 1)):
      logging.info('Evaluating.')
      eval_actor.run_and_log()
      last_eval_step = agent_learner.train_step_numpy

  rb_observer.close()
  reverb_server.stop()
