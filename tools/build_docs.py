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

r"""Generate docs for TF-Agents.

# How to run

```
python build_docs.py --output_dir=/path/to/output
```

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from absl import app
from absl import flags

import tensorflow as tf

from tensorflow_docs.api_generator import doc_controls
from tensorflow_docs.api_generator import generate_lib
from tensorflow_docs.api_generator import public_api

import tf_agents
# pylint: disable=unused-import
from tf_agents import agents
from tf_agents import distributions
from tf_agents import drivers
from tf_agents import environments
from tf_agents import metrics
from tf_agents import networks
from tf_agents import policies
from tf_agents import replay_buffers
from tf_agents import specs
from tf_agents import trajectories
from tf_agents import utils
# pylint: enable=unused-import

flags.DEFINE_string('output_dir', '/tmp/agents_api/',
                    'The path to output the files to')

flags.DEFINE_string('code_url_prefix',
                    'https://github.com/tensorflow/agents/blob/master/',
                    'The url prefix for links to code.')

flags.DEFINE_bool('search_hints', True,
                  'Include metadata search hints in the generated files')

flags.DEFINE_string('site_path', 'agents/api_docs/python',
                    'Path prefix in the _toc.yaml')

FLAGS = flags.FLAGS


def main(_):
  for cls in [tf.Module, tf.keras.layers.Layer]:
    doc_controls.decorate_all_class_attributes(
        decorator=doc_controls.do_not_doc_in_subclasses,
        cls=cls,
        skip=['__init__'])

  doc_generator = generate_lib.DocGenerator(
      root_title='TF-Agents',
      py_modules=[('tf_agents', tf_agents)],
      base_dir=os.path.dirname(tf_agents.__file__),
      code_url_prefix=FLAGS.code_url_prefix,
      search_hints=FLAGS.search_hints,
      site_path=FLAGS.site_path,
      private_map={},
      callbacks=[public_api.local_definitions_filter])

  sys.exit(doc_generator.build(output_dir=FLAGS.output_dir))


if __name__ == '__main__':
  app.run(main)
