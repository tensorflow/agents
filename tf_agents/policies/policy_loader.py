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
"""Helper function to load policies from disk."""
import os
from typing import Optional, Text

import tensorflow.compat.v2 as tf
from tf_agents.policies import policy_saver
from tf_agents.policies import py_tf_eager_policy


def load(saved_model_path: Text, checkpoint_path: Optional[Text] = None):
  """Loads a policy.

  The argument `saved_model_path` is the path of a directory containing a full
  saved model for the policy. The path typically looks like
  '/root_dir/policies/policy', it may contain trailing numbers for the
  train_step.

  `saved_model_path` is expected to contain the following files. (There can be
  additional shards for the `variables.data` files.)
     * `saved_model.pb`
     * `policy_specs.pbtxt`
     * `variables/variables.index`
     * `variables/variables.data-00000-of-00001`

  The optional argument `checkpoint_path` is the path to a directory that
  contains variable checkpoints (as opposed to full saved models) for the
  policy. The path also typically ends-up with the checkpoint number,
  for example: '/my/save/dir/checkpoint/000022100'.

  If specified, `checkpoint_path` is expected to contain the following
  files. (There can be additional shards for the `variables.data` files.)
     * `variables/variables.index`
     * `variables/variables.data-00000-of-00001`

  `load()` recreates a policy from the saved model, and if it was specified
  updates the policy from the checkpoint.  It returns the policy.

  Args:
    saved_model_path: string. Path to a directory containing a full saved model.
    checkpoint_path: string. Optional path to a directory containing a
      checkpoint of the model variables.

  Returns:
    A `tf_agents.policies.SavedModelPyTFEagerPolicy`.
  """
  policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
      saved_model_path, load_specs_from_pbtxt=True)
  if checkpoint_path:
    policy.update_from_checkpoint(checkpoint_path)
  return policy


def _copy_file(from_dir, name, to_dir):
  tf.io.gfile.copy(os.path.join(from_dir, name), os.path.join(to_dir, name))


def _copy_dir(from_dir, name, to_dir):
  from_dir_name = os.path.join(from_dir, name)
  to_dir_name = os.path.join(to_dir, name)
  tf.io.gfile.mkdir(to_dir_name)
  for file_name in tf.io.gfile.listdir(from_dir_name):
    _copy_file(from_dir_name, file_name, to_dir_name)


def materialize_saved_model(saved_model_path: Text, checkpoint_path: Text,
                            output_path: Text):
  """Materializes a full saved model for a policy.

  Some training processes generate a full saved model only at step 0, and then
  generate checkpoints for the model variables at different train steps. In this
  case there are no full saved models available for these train steps and you
  must pass both the path to initial full saved model and the path to the
  checkpoint to load a model at a given train step.

  This function allows you to assemble a full saved model by combining the full
  saved model at step 0 with a checkpoint at a further train step. The new saved
  model is all that is needed to deploy the model for testing or production.

  The arguments `saved_model_path` and `checkpoint_path` are exactly as for
  `load()`:

     * `saved_model_path` is the path to the full saved model at step 0.
     * `checkpoint_path` is the path to the variable checkpoint at a further
       step.

  `output_path` must be a non-existent path on disk. It will be created as a new
  directory. After `materialize_saved_model()` runs `output_path` will contain
  the following files (There can be additional shards for the `variables.data`
  files.)

     * `saved_model.pb`
     * `policy_specs.pbtxt`
     * `variables/variables.index`
     * `variables/variables.data-00000-of-00001`:

  After running this function you can pass `output_path` to
  `policy_loader.load()` to load the policy.

  Example usage:

  ```python
  # The training process generated a saved model at
  # `/path/policies/collect_policy` and checkpoints at
  # `/path/policies/checkpoint/policy_checkpoint_NNNNNNNN`
  #
  # Assemble in '/path/policies/collect_policy/prod' a full saved model
  # with the checkpoint at step 13400:
  policy_loader.materialize_saved_model(
      '/path/policies/collect_policy',
      '/path/policies/checkpoint/policy_checkpoint_0001340',
      '/path/policies/collect_policy/prod')
  ...
  # Later, load a model from the assembled model
  collect_policy = policy_loader.load('/path/policies/collect_policy/prod')
  ```

  Args:
    saved_model_path: string. Path to a directory containing a full saved model.
    checkpoint_path: string. Path to a directory containing a checkpoint of the
      model variables.
    output_path: string. Path where to save the materialized full saved model.
  """
  if tf.io.gfile.exists(output_path):
    raise ValueError('Output path already exists: %s' % output_path)
  tf.io.gfile.makedirs(output_path)
  _copy_dir(checkpoint_path, tf.saved_model.VARIABLES_DIRECTORY, output_path)
  _copy_dir(saved_model_path, tf.saved_model.ASSETS_DIRECTORY, output_path)
  _copy_file(saved_model_path, tf.saved_model.SAVED_MODEL_FILENAME_PB,
             output_path)
  _copy_file(saved_model_path, policy_saver.POLICY_SPECS_PBTXT, output_path)
