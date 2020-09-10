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

# Lint as: python3
r"""Train and Eval PPOClipAgent in the Mujoco environments.

All hyperparameters come from the PPO paper
https://arxiv.org/abs/1707.06347.pdf
"""
import os

from absl import app
from absl import flags
from absl import logging

import tensorflow.compat.v2 as tf

from tf_agents.experimental.examples.ppo import train_eval_lib

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


def main(_):
  logging.set_verbosity(logging.INFO)
  tf.enable_v2_behavior()

  train_eval_lib.train_eval(
      FLAGS.root_dir,
      env_name='HalfCheetah-v2',
      # Training params
      num_iterations=FLAGS.num_iterations,
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
      # Replay params
      reverb_port=FLAGS.reverb_port,
      replay_capacity=10000,
      # Others
      policy_save_interval=5000,
      summary_interval=1000,
      eval_interval=FLAGS.eval_interval)


if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  app.run(main)
