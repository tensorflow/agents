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

r"""Train and Eval PPO Clip Agent, with required atari import.

Launch train eval binary:

For usage, see train_eval_clip_agent.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.agents.ppo.examples.v1 import train_eval_clip_agent
from tf_agents.environments import suite_atari
from tf_agents.system import system_multiprocessing as multiprocessing

FLAGS = flags.FLAGS


def main(_):
  tf.compat.v1.enable_resource_variables()
  logging.set_verbosity(logging.INFO)
  train_eval_clip_agent.train_eval(
      FLAGS.root_dir,
      tf_master=FLAGS.master,
      env_name=FLAGS.env_name,
      env_load_fn=suite_atari.load,
      replay_buffer_capacity=FLAGS.replay_buffer_capacity,
      num_environment_steps=FLAGS.num_environment_steps,
      num_parallel_environments=FLAGS.num_parallel_environments,
      num_epochs=FLAGS.num_epochs,
      collect_episodes_per_iteration=FLAGS.collect_episodes_per_iteration,
      num_eval_episodes=FLAGS.num_eval_episodes)


if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  multiprocessing.handle_main(lambda _: app.run(main))
