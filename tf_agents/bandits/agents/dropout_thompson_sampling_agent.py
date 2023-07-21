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

"""A neural network based agent that implements Thompson sampling via dropout.

Implements an agent based on a neural network that predicts arm rewards.
The neural network internally uses dropout to approximate Thompson sampling.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Iterable, Optional, Text, Tuple

import gin
import tensorflow as tf

from tf_agents.bandits.agents import greedy_reward_prediction_agent
from tf_agents.bandits.networks import heteroscedastic_q_network
from tf_agents.bandits.policies import constraints as constr
from tf_agents.networks import q_network
from tf_agents.typing import types


# TODO(b/146206372): refactor DropoutThompsonSamplingAgent API to be compliant
# with other APIs which take a reward network at initialisation
@gin.configurable
class DropoutThompsonSamplingAgent(
    greedy_reward_prediction_agent.GreedyRewardPredictionAgent):
  """A neural network based Thompson sampling agent.

  This agent receives parameters for a neural network and trains it to predict
  rewards. The action is chosen greedily with respect to the prediction.
  The neural network implements dropout for exploration.
  """

  def __init__(
      self,
      time_step_spec: types.TimeStep,
      action_spec: types.BoundedTensorSpec,
      optimizer: types.Optimizer,
      # Network params.
      dropout_rate: types.FloatOrReturningFloat,
      network_layers: Tuple[int, ...],
      dropout_only_top_layer: bool = True,
      observation_and_action_constraint_splitter: Optional[
          types.Splitter] = None,
      constraints: Iterable[constr.BaseConstraint] = (),
      # Params for training.
      error_loss_fn: types.LossFn = tf.compat.v1.losses.mean_squared_error,
      gradient_clipping: Optional[float] = None,
      heteroscedastic: bool = False,
      # Params for debugging.
      debug_summaries: bool = False,
      summarize_grads_and_vars: bool = False,
      enable_summaries: bool = True,
      emit_policy_info: Tuple[Text, ...] = (),
      train_step_counter: Optional[tf.Variable] = None,
      laplacian_matrix: Optional[types.Float] = None,
      laplacian_smoothing_weight: float = 0.001,
      name: Optional[Text] = None):
    """Creates a Dropout Thompson Sampling Agent.

    For more details about the Laplacian smoothing regularization, please see
    the documentation of the `GreedyRewardPredictionAgent`.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of `BoundedTensorSpec` representing the actions.
      optimizer: The optimizer to use for training.
      dropout_rate: Float in `(0, 1)`, or a function that takes no arguments and
        returns a float in `(0,1)`. It represents the dropout rate used by the
        model.
      network_layers: Tuple of ints determining the sizes of the network layers.
      dropout_only_top_layer: Boolean parameter determining if dropout should be
        done only in the top layer. True by default.
      observation_and_action_constraint_splitter: A function used for masking
        valid/invalid actions with each state of the environment. The function
        takes in a full observation and returns a tuple consisting of 1) the
        part of the observation intended as input to the bandit agent and
        policy, and 2) the boolean mask. This function should also work with a
        `TensorSpec` as input, and should output `TensorSpec` objects for the
        observation and mask.
      constraints: iterable of constraints objects that are instances of
        `tf_agents.bandits.agents.NeuralConstraint`.
      error_loss_fn: A function for computing the error loss, taking parameters
        labels, predictions, and weights (any function from tf.losses would
        work). The default is `tf.losses.mean_squared_error`.
      gradient_clipping: A float representing the norm length to clip gradients
        (or None for no clipping.)
      heteroscedastic: If True, the variance per action is estimated and the
        losses are weighted appropriately.
      debug_summaries: A Python bool, default False. When True, debug summaries
        are gathered.
      summarize_grads_and_vars: A Python bool, default False. When True,
        gradients and network variable summaries are written during training.
      enable_summaries: A Python bool, default True. When False, all summaries
        (debug or otherwise) should not be written.
      emit_policy_info: (tuple of strings) what side information we want to get
        as part of the policy info. Allowed values can be found in
        `policy_utilities.PolicyInfo`.
      train_step_counter: An optional `tf.Variable` to increment every time the
        train op is run.  Defaults to the `global_step`.
      laplacian_matrix: A float `Tensor` shaped `[num_actions, num_actions]`.
        This holds the Laplacian matrix used to regularize the smoothness of the
        estimated expected reward function. This only applies to problems where
        the actions have a graph structure. If `None`, the regularization is not
        applied.
      laplacian_smoothing_weight: A float that determines the weight of the
        regularization term. Note that this has no effect if `laplacian_matrix`
        above is `None`.
      name: Python str name of this agent. All variables in this module will
        fall under that name. Defaults to the class name.

    Raises:
      ValueError: If the action spec contains more than one action or or it is
      not a bounded scalar int32 spec with minimum 0.
    """
    self._dropout_rate = dropout_rate
    fc_layer_params = network_layers
    dropout_param = {'rate': dropout_rate, 'permanent': True}
    if dropout_only_top_layer:
      dropout_layer_params = [None] * (len(fc_layer_params) - 1)
      dropout_layer_params.append(dropout_param)
    else:
      dropout_layer_params = [dropout_param] * len(fc_layer_params)
    if observation_and_action_constraint_splitter is not None:
      input_tensor_spec, _ = observation_and_action_constraint_splitter(
          time_step_spec.observation)
    else:
      input_tensor_spec = time_step_spec.observation

    if heteroscedastic:
      reward_network = heteroscedastic_q_network.HeteroscedasticQNetwork(
          input_tensor_spec=input_tensor_spec,
          action_spec=action_spec,
          fc_layer_params=fc_layer_params,
          dropout_layer_params=dropout_layer_params)
    else:
      reward_network = q_network.QNetwork(
          input_tensor_spec=input_tensor_spec,
          action_spec=action_spec,
          fc_layer_params=fc_layer_params,
          dropout_layer_params=dropout_layer_params)

    super(DropoutThompsonSamplingAgent, self).__init__(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        reward_network=reward_network,
        optimizer=optimizer,
        observation_and_action_constraint_splitter=(
            observation_and_action_constraint_splitter),
        constraints=constraints,
        error_loss_fn=error_loss_fn,
        gradient_clipping=gradient_clipping,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        enable_summaries=enable_summaries,
        emit_policy_info=emit_policy_info,
        train_step_counter=train_step_counter,
        laplacian_matrix=laplacian_matrix,
        laplacian_smoothing_weight=laplacian_smoothing_weight,
        name=name)

  def _train(self, experience, weights):
    if callable(self._dropout_rate):
      dropout = self._dropout_rate()
    else:
      dropout = self._dropout_rate
    tf.compat.v2.summary.scalar(
        name='dropout', data=dropout, step=self.train_step_counter)
    return super(DropoutThompsonSamplingAgent, self)._train(experience, weights)
