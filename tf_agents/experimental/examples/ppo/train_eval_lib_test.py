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

"""Tests for train_eval_lib."""

from tf_agents.experimental.examples.ppo import train_eval_lib
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.utils import test_utils


class TrainEvalLibTest(test_utils.TestCase):

  def test_train_eval(self):
    train_eval_lib.train_eval(
        root_dir=self.create_tempdir(),
        env_name='HalfCheetah-v2',
        # Training params
        num_iterations=2,
        actor_fc_layers=(20, 10),
        value_fc_layers=(20, 10),
        learning_rate=1e-3,
        collect_sequence_length=10,
        minibatch_size=None,
        num_epochs=2,
        # Agent params
        importance_ratio_clipping=0.2,
        lambda_value=0.95,
        discount_factor=0.99,
        entropy_regularization=0.,
        value_pred_loss_coef=0.5,
        use_gae=True,
        use_td_lambda_return=True,
        gradient_clipping=None,
        value_clipping=None,
        # Replay params
        reverb_port=None,
        replay_capacity=10000,
        # Others
        policy_save_interval=0,
        summary_interval=0,
        eval_interval=0)


if __name__ == '__main__':
  multiprocessing.handle_test_main(test_utils.main)
