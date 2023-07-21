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

"""TF Agents."""

# We need to put some imports inside a function call below, and the function
# call needs to come before the *actual* imports that populate the
# tf_agents namespace. Hence, we disable this lint check throughout
# the file.
#
# pylint: disable=g-import-not-at-top


# Ensure TensorFlow is importable and its version is sufficiently recent. This
# needs to happen before anything else, since the imports below will try to
# import tensorflow, too.
def _ensure_tf_install():  # pylint: disable=g-statement-before-imports
  """Attempt to import tensorflow, and ensure its version is sufficient.

  Raises:
    ImportError: if either tensorflow is not importable or its version is
    inadequate.
  """
  try:
    import tensorflow as tf
  except (ImportError, ModuleNotFoundError):
    # Print more informative error message, then reraise.
    print("\n\nFailed to import TensorFlow. Please note that TensorFlow is not "
          "installed by default when you install TF Agents. This is so that "
          "users can decide whether to install the GPU-enabled TensorFlow "
          "package. To use TF Agents, please install the most recent version "
          "of TensorFlow, by following instructions at "
          "https://tensorflow.org/install.\n\n")
    raise

  import distutils.version

  #
  # Update this whenever we need to depend on a newer TensorFlow release.
  #
  required_tensorflow_version = "2.2.0"

  tf_version = tf.version.VERSION
  if (distutils.version.LooseVersion(tf_version) <
      distutils.version.LooseVersion(required_tensorflow_version)):
    raise ImportError(
        "This version of TF Agents requires TensorFlow "
        "version >= {required}; Detected an installation of version {present}. "
        "Please upgrade TensorFlow to proceed.".format(
            required=required_tensorflow_version,
            present=tf_version))


_ensure_tf_install()

import sys as _sys

from tf_agents import agents
from tf_agents import bandits
from tf_agents import distributions
from tf_agents import drivers
from tf_agents import environments
from tf_agents import eval  # pylint: disable=redefined-builtin
from tf_agents import experimental
from tf_agents import keras_layers
from tf_agents import metrics
from tf_agents import networks
from tf_agents import policies
from tf_agents import replay_buffers
from tf_agents import specs
from tf_agents import system
from tf_agents import train
from tf_agents import trajectories
from tf_agents import typing
from tf_agents import utils
from tf_agents import version

from tf_agents.version import __version__

# Cleanup symbols to avoid polluting namespace.
for symbol in ["_ensure_tf_install", "_sys"]:
  delattr(_sys.modules[__name__], symbol)
