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

"""Module for numpy array and `tf.Tensor` shape and dtype specifications."""

# TODO(b/130564501): Do not import classes directly, only expose
# modules.
from tf_agents.specs import array_spec
from tf_agents.specs import bandit_spec_utils
from tf_agents.specs import distribution_spec
from tf_agents.specs import tensor_spec

from tf_agents.specs.array_spec import ArraySpec
from tf_agents.specs.array_spec import BoundedArraySpec

from tf_agents.specs.tensor_spec import BoundedTensorSpec
from tf_agents.specs.tensor_spec import from_spec
from tf_agents.specs.tensor_spec import is_bounded
from tf_agents.specs.tensor_spec import is_continuous
from tf_agents.specs.tensor_spec import is_discrete
from tf_agents.specs.tensor_spec import sample_spec_nest
from tf_agents.specs.tensor_spec import TensorSpec
from tf_agents.specs.tensor_spec import zero_spec_nest
