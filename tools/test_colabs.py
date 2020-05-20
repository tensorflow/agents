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

"""Tests colabs using Jupyter notebook."""
from __future__ import print_function

import os
import sys

from absl import app
from absl import flags
from absl import logging

from nbconvert.preprocessors import CellExecutionError
from nbconvert.preprocessors import ExecutePreprocessor
import nbformat

flags.DEFINE_string('output_dir', '/tmp/notebook_tests',
                    'Full path for executed notebooks and artifacts.')
flags.DEFINE_boolean('debug', True,
                     'Debug logging if true. Otherwise info only.')
flags.DEFINE_boolean('override_pip_install_agents', True,
                     'If true a replace is done to prevent notebooks from '
                     'installing tf-agents (often tf-agents-nightly)')
FLAGS = flags.FLAGS


def execute_test(file_path, result_path):
  """Executes a single notebook.

  Args:
    file_path: Path to the notebook to execute.
    result_path: Path to store the resulting notebook.

  Returns:
    bool: True if the notebook does not have any errors, False otherwise.

  Raises:
    Exception if an unexpected error occurs executing the notebook.
  """
  try:
    with open(file_path, 'r') as f:
      filedata = f.read()
      if FLAGS.override_pip_install_agents:
        filedata = filedata.replace('pip install tf-agents', 'pip --version')
      nb = nbformat.reads(filedata, as_version=4)

      ep = ExecutePreprocessor(timeout=3600, kernel_name='python3')
      try:
        ep.preprocess(nb, {'metadata': {'path': FLAGS.output_dir}})
      except CellExecutionError as cex:
        logging.error('ERROR executing:%s', file_path)
        logging.error(cex)
        return False
    with open(result_path, 'w', encoding='utf-8') as fo:
      nbformat.write(nb, fo)
    return True
  except Exception as e:  # pylint: disable=W0703
    logging.error('Unexpected ERROR: in %s', file_path)
    logging.error(e)


def run():
  """Runs all notebooks and reports results."""
  os.makedirs(FLAGS.output_dir)
  colab_path = './docs/tutorials/'
  _, _, filenames = next(os.walk(colab_path))

  passed = []
  failed = []
  filenames.sort()
  for filename in filenames:
    logging.info('Testing %s ...', filename)
    if 'ipynb' not in filename:
      logging.debug('Skipping non-notebook file:%s', filename)
      continue
    if '7_SAC_minitaur_tutorial.ipynb' in filename:
      logging.info('Skipping 7_SAC_minitaur_tutorial.ipynb. '
                   'It takes 8 hours to run.')
      continue
    file_path = os.path.join(colab_path, filename)
    result_path = os.path.join(FLAGS.output_dir, 'executed_' + filename)
    if execute_test(file_path, result_path):
      passed.append(filename)
    else:
      failed.append(filename)

  logging.info('\n\n################# Report #################')
  logging.info('%d passed, %d failed', len(passed), len(failed))
  for p_result in passed:
    logging.info('%s OK', p_result)
  for f_result in failed:
    logging.info('%s FAILED', f_result)

  if failed:
    sys.exit(1)


def main(_):
  logging.set_verbosity(logging.INFO)
  if FLAGS.debug:
    logging.set_verbosity(logging.DEBUG)
  run()


if __name__ == '__main__':
  app.run(main)
