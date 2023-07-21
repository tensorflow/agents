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

"""Utils for processing specs."""

from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory


def get_tensor_specs(env):
  """Returns observation, action and time step TensorSpecs from passed env.

  Args:
    env: environment instance used for collection.
  """
  observation_tensor_spec = tensor_spec.from_spec(env.observation_spec())
  action_tensor_spec = tensor_spec.from_spec(env.action_spec())
  time_step_tensor_spec = ts.time_step_spec(observation_tensor_spec)

  return observation_tensor_spec, action_tensor_spec, time_step_tensor_spec


def get_collect_data_spec_from_policy_and_env(env, policy):
  """Returns collect data spec from policy and environment.

  Args:
    env: instance of the environment used for collection
    policy: policy for collection to get policy spec

  Meant to be used for collection jobs (i.e. Actors) without having to
  construct an agent instance but directly from a policy (which can be loaded
  from a saved model).
  """
  observation_tensor_spec = tensor_spec.from_spec(env.observation_spec())
  time_step_tensor_spec = ts.time_step_spec(observation_tensor_spec)
  policy_step_tensor_spec = tensor_spec.from_spec(
      policy.policy_step_spec)
  collect_data_spec = trajectory.from_transition(time_step_tensor_spec,
                                                 policy_step_tensor_spec,
                                                 time_step_tensor_spec)
  return collect_data_spec
