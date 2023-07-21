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

r"""Suite for loading pybullet Gym environments.

Importing pybullet_envs registers the environments. Once this is done the
regular gym loading mechanism used in suite_gym will generate pybullet envs.

For a list of registered pybullet environments take a look at:
  pybullet_envs/__init__.py

To visualize a pybullet environment as it is being run you can launch the
example browser BEFORE you start the training.

```bash
ExampleBrowser -- --start_demo_name="PhysicsServer"
```
"""
import gin
from tf_agents.environments import suite_gym

# pylint: disable=unused-import
import pybullet_envs
# pylint: enable=unused-import

load = gin.external_configurable(suite_gym.load, 'suite_pybullet.load')
