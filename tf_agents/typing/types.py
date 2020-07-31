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

"""Common types used in TF-Agents."""

from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

import sys
import typing
from typing import Callable, Iterable, Mapping, Optional, Sequence, Text, TypeVar, Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.specs import array_spec

if sys.version_info < (3, 7):
  ForwardRef = typing._ForwardRef  # pylint: disable=protected-access
else:
  ForwardRef = typing.ForwardRef

Tensor = Union[tf.Tensor, tf.SparseTensor, tf.RaggedTensor]
Array = np.ndarray   # pylint: disable=invalid-name
TensorOrArray = Union[Tensor, Array]
Distribution = tfp.distributions.Distribution

TensorSpec = tf.TypeSpec
ArraySpec = array_spec.ArraySpec
Spec = Union[TensorSpec, ArraySpec]

SpecTensorOrArray = Union[Spec, Tensor, Array]

Network = ForwardRef('tf_agents.networks.network.Network')  # pylint: disable=invalid-name

# Note that this is effectively treated as `Any`; see b/109648354.
Tnest = TypeVar('Tnest')
Nested = Union[Tnest, Iterable[Tnest], Mapping[Text, Tnest]]
NestedTensor = Nested[Tensor]
NestedVariable = Nested[tf.Variable]
NestedArray = Nested[Array]
NestedDistribution = Nested[tfp.distributions.Distribution]
NestedPlaceHolder = Nested[tf.compat.v1.placeholder]
NestedTensorSpec = Nested[TensorSpec]
NestedArraySpec = Nested[array_spec.ArraySpec]
NestedLayer = Nested[tf.keras.layers.Layer]
NestedNetwork = Nested[Network]

NestedSpec = Union[NestedTensorSpec, NestedArraySpec]
NestedTensorOrArray = Union[NestedTensor, NestedArray]
NestedSpecTensorOrArray = Union[NestedSpec, NestedTensor, NestedArray]

Int = Union[int, np.int16, np.int32, np.int64, Tensor, Array]
Bool = Union[bool, np.bool, Tensor, Array]

Float = Union[float, np.float16, np.float32, np.float64, Tensor, Array]
FloatOrReturningFloat = Union[Float, Callable[[], Float]]

Shape = Union[TensorOrArray, Sequence[Optional[int]], tf.TensorShape]

Splitter = Optional[Callable[
    [NestedSpecTensorOrArray], Iterable[NestedSpecTensorOrArray]]]
Seed = Union[int, Sequence[int], Tensor, Array]

TimeStep = ForwardRef('tf_agents.trajectories.time_step.TimeStep')  # pylint: disable=invalid-name
PolicyStep = ForwardRef('tf_agents.trajectories.policy_step.PolicyStep')  # pylint: disable=invalid-name

GymEnv = ForwardRef('gym.Env')  # pylint: disable=invalid-name
GymEnvWrapper = Callable[[GymEnv], GymEnv]

PyEnv = ForwardRef('tf_agents.environments.py_environment.PyEnvironment')  # pylint: disable=invalid-name
PyEnvWrapper = Callable[[PyEnv], PyEnv]

LossFn = Callable[[Tensor, Tensor], Tensor]

Optimizer = Union[tf.keras.optimizers.Optimizer, tf.compat.v1.train.Optimizer]

# We use lazy loading of Reverb, so we predeclare common Reverb objects
ReverbServer = ForwardRef('reverb.Server')
ReverbTable = ForwardRef('reverb.Table')
ReverbClient = ForwardRef('reverb.Client')
ReverbTFClient = ForwardRef('reverb.TFClient')
