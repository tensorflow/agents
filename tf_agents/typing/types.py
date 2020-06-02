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
from typing import Callable, Iterable, Mapping, Optional, Sequence, Text, Tuple, Union

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

# Note that this is effectively treated as `Any`; see b/109648354.

# pytype: disable=not-supported-yet
NestedTensor = Union[Tensor, Iterable['NestedTensor'],
                     Mapping[str, 'NestedTensor']]
NestedVariable = Union[tf.Variable, Iterable['NestedVariable'],
                       Mapping[str, 'NestedVariable']]
NestedArray = Union[Array, Iterable['NestedArray'],
                    Mapping[str, 'NestedArray']]
NestedDistribution = Union[tfp.distributions.Distribution,
                           Iterable['NestedDistribution'],
                           Mapping[str, 'NestedDistribution']]
NestedPlaceHolder = Union[tf.compat.v1.placeholder,
                          Iterable['NestedPlaceHolder'],
                          Mapping[Text, 'NestedPlaceHolder']]

NestedTensorSpec = Union[TensorSpec, Iterable['NestedTensorSpec'],
                         Mapping[str, 'NestedTensorSpec']]
NestedArraySpec = Union[array_spec.ArraySpec, Iterable['NestedArraySpec'],
                        Mapping[str, 'NestedArraySpec']]
# pytype: enable=not-supported-yet

NestedSpec = Union[NestedTensorSpec, NestedArraySpec]
NestedTensorOrArray = Union[NestedTensor, NestedArray]
NestedSpecTensorOrArray = Union[NestedSpec, NestedTensor, NestedArray]

Int = Union[int, np.int16, np.int32, np.int64, Tensor, Array]
Bool = Union[bool, np.bool, Tensor, Array]

Float = Union[float, np.float16, np.float32, np.float64, Tensor, Array]
FloatOrReturningFloat = Union[Float, Callable[[], Float]]

Shape = Union[TensorOrArray, Sequence[int], tf.TensorShape]

Splitter = Optional[Callable[
    [NestedSpecTensorOrArray], Iterable[NestedSpecTensorOrArray]]]
Seed = Union[int, Sequence[int], Tensor, Array]

TimeStep = ForwardRef('tf_agents.trajectories.TimeStep')  # pylint: disable=invalid-name
PolicyStep = ForwardRef('tf_agents.trajectories.PolicyStep')  # pylint: disable=invalid-name
Transition = Tuple[TimeStep, PolicyStep, TimeStep]

GymEnv = ForwardRef('gym.Env')  # pylint: disable=invalid-name
GymEnvWrapper = Callable[[GymEnv], GymEnv]

PyEnv = ForwardRef('tf_agents.environments.PyEnvironment')  # pylint: disable=invalid-name
PyEnvWrapper = Callable[[PyEnv], PyEnv]

LossFn = Callable[[Tensor, Tensor], Tensor]

Optimizer = Union[tf.keras.optimizers.Optimizer, tf.compat.v1.train.Optimizer]
