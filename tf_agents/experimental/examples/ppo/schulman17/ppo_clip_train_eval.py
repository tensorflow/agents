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
r"""Train and Eval PPOClipAgent in the Mujoco environments.

All hyperparameters come from the PPO paper
https://arxiv.org/abs/1707.06347.pdf
"""
import os

from absl import app
from absl import flags
from absl import logging

import gin

import tensorflow.compat.v2 as tf

from tf_agents.experimental.examples.ppo.schulman17 import train_eval_lib

FLAGS = flags.FLAGS

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_integer(
    'reverb_port', None,
    'Port for reverb server, if None, use a randomly chosen unused port.')
flags.DEFINE_integer('num_iterations', 1600,
                     'Total number train/eval iterations to perform.')
flags.DEFINE_integer(
    'eval_interval', 10000,
    'Number of train steps between evaluations. Set to 0 to skip.')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin binding parameters.')


def ppo_clip_train_eval(root_dir, num_iterations, reverb_port=None,
                        eval_interval=0):
  """Executes train and eval for ppo_clip.

  gin is used to configure parameters related to the agent and environment.
  Arguments related to the execution, e.g. number of iterations and how often to
  eval, are set directly by this method. This keeps the gin config focused on
  the agent and execution level arguments quickly changed on the command line
  without using the more verbose --gin_bindings.

  See the `./configs` directory for example gin configs.

  Args:
    root_dir: Root directory for writing logs/summaries/checkpoints.
    num_iterations: Total number train/eval iterations to perform.
    reverb_port: Port for reverb server. If None, a random unused port is used.
    eval_interval: Number of train steps between evaluations. Set to 0 to skip.
  """
  train_eval_lib.train_eval(
      root_dir,
      # Training params
      num_iterations=num_iterations,
      # Replay params
      reverb_port=reverb_port,
      # Others
      eval_interval=eval_interval)


def main(_):
  logging.set_verbosity(logging.INFO)
  tf.enable_v2_behavior()

  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_bindings)

  ppo_clip_train_eval(FLAGS.root_dir,
                      num_iterations=FLAGS.num_iterations,
                      reverb_port=FLAGS.reverb_port,
                      eval_interval=FLAGS.eval_interval)


if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  app.run(main)
