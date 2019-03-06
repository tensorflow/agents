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

r"""Train and Eval PPO, with required atari import.

Launch train eval binary:

For usage, see train_eval.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import numpy as np
import tensorflow as tf

from tf_agents.agents.ppo.examples.v1 import train_eval
from tf_agents.environments import random_py_environment
from tf_agents.specs import array_spec

FLAGS = flags.FLAGS


def env_load_fn(env_name):
  del env_name
  obs_spec = array_spec.BoundedArraySpec((2,), np.int32, -10, 10)
  action_spec = array_spec.BoundedArraySpec((2, 3), np.int32, -10, 10)
  return random_py_environment.RandomPyEnvironment(
      obs_spec, action_spec=action_spec, min_duration=2, max_duration=4)


def main(_):
  tf.compat.v1.enable_resource_variables()
  if tf.executing_eagerly():
    # self.skipTest('b/123777119')  # Secondary bug: ('b/123775375')
    return
  logging.set_verbosity(logging.INFO)
  train_eval.train_eval(
      FLAGS.root_dir, tf_master=FLAGS.master,
      env_name=FLAGS.env_name,
      env_load_fn=env_load_fn,
      replay_buffer_capacity=FLAGS.replay_buffer_capacity,
      num_environment_steps=FLAGS.num_environment_steps,
      num_parallel_environments=FLAGS.num_parallel_environments,
      num_epochs=FLAGS.num_epochs,
      collect_episodes_per_iteration=FLAGS.collect_episodes_per_iteration,
      num_eval_episodes=FLAGS.num_eval_episodes,
      use_rnns=FLAGS.use_rnns)


if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  app.run(main)
