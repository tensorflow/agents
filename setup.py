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

"""Build, test, and install tf_agents."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import codecs
import datetime
import fnmatch
import io
import os
import subprocess
import sys
import unittest

from setuptools import find_packages
from setuptools import setup
from setuptools.command.test import test as TestCommandBase
from setuptools.dist import Distribution

# To enable importing version.py directly, we add its path to sys.path.
version_path = os.path.join(os.path.dirname(__file__), 'tf_agents')
sys.path.append(version_path)
import version as tf_agents_version  # pylint: disable=g-import-not-at-top

# Default versions for packages we often override for testing and release
# candidates. These can all be overridden with flags.
TFP_VERSION = 'tensorflow-probability'
TFP_NIGHTLY = 'tfp-nightly'
TENSORFLOW_VERSION = 'tensorflow'
TENSORFLOW_NIGHTLY = 'tf-nightly'
REVERB_VERSION = 'dm-reverb'
REVERB_NIGHTLY = 'dm-reverb-nightly'
RLDS_VERSION = 'rlds'
# TODO(b/224850217): rlds does not have nightly builds yet.
RLDS_NIGHTLY = 'rlds'


class StderrWrapper(io.IOBase):

  def write(self, *args, **kwargs):
    return sys.stderr.write(*args, **kwargs)

  def writeln(self, *args, **kwargs):
    if args or kwargs:
      sys.stderr.write(*args, **kwargs)
    sys.stderr.write('\n')


class TestLoader(unittest.TestLoader):

  def __init__(self, exclude_list):
    super(TestLoader, self).__init__()
    self._exclude_list = exclude_list

  def _match_path(self, path, full_path, pattern):
    if not fnmatch.fnmatch(path, pattern):
      return False
    module_name = full_path.replace('/', '.').rstrip('.py')
    if any(module_name.endswith(x) for x in self._exclude_list):
      return False
    return True


def load_test_list(filename):
  testcases = [x.rstrip() for x in open(filename, 'r').readlines() if x]
  # Remove comments and blanks after comments are removed.
  testcases = [x.partition('#')[0].strip() for x in testcases]
  return [x for x in testcases if x]


class Test(TestCommandBase):

  def run_tests(self):
    # Import absl inside run, where dependencies have been loaded already.
    from absl import app  # pylint: disable=g-import-not-at-top

    def main(_):
      # pybullet imports multiprocessing in their setup.py, which causes an
      # issue when we import multiprocessing.pool.dummy down the line because
      # the PYTHONPATH has changed.
      for module in [
          'multiprocessing', 'multiprocessing.pool', 'multiprocessing.dummy',
          'multiprocessing.pool.dummy'
      ]:
        if module in sys.modules:
          del sys.modules[module]
      # Reimport multiprocessing to avoid spurious error printouts. See
      # https://bugs.python.org/issue15881.
      import multiprocessing as _  # pylint: disable=g-import-not-at-top
      import tensorflow as tf  # pylint: disable=g-import-not-at-top

      # Sets all GPUs to 1GB of memory. The process running the bulk of the unit
      # tests allocates all GPU memory because by default TensorFlow allocates
      # all GPU memory during initialization. This causes tests in
      # run_seperately to fail with out of memory errors because they are run as
      # a subprocess of the process holding the GPU memory.
      gpus = tf.config.experimental.list_physical_devices('GPU')
      for gpu in gpus:
        tf.config.set_logical_device_configuration(
            gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])

      run_separately = load_test_list('test_individually.txt')
      broken_tests = load_test_list(FLAGS.broken_tests)

      test_loader = TestLoader(exclude_list=run_separately + broken_tests)
      test_suite = test_loader.discover('tf_agents', pattern='*_test.py')
      stderr = StderrWrapper()
      result = unittest.TextTestResult(stderr, descriptions=True, verbosity=2)
      test_suite.run(result)

      external_test_failures = []

      for test in run_separately:
        filename = 'tf_agents/%s.py' % test.replace('.', '/')
        try:
          subprocess.check_call([sys.executable, filename])
        except subprocess.CalledProcessError as e:
          external_test_failures.append(e)

      result.printErrors()

      for failure in external_test_failures:
        stderr.writeln(str(failure))

      final_output = (
          'Tests run: {} grouped and {} external.  '.format(
              result.testsRun, len(run_separately)) +
          'Errors: {}  Failures: {}  External failures: {}.'.format(
              len(result.errors),
              len(result.failures),
              len(external_test_failures)))

      header = '=' * len(final_output)
      stderr.writeln(header)
      stderr.writeln(final_output)
      stderr.writeln(header)

      if result.wasSuccessful() and not external_test_failures:
        return 0
      else:
        return 1

    # Run inside absl.app.run to ensure flags parsing is done.
    from tf_agents.system import system_multiprocessing as multiprocessing  # pylint: disable=g-import-not-at-top
    return multiprocessing.handle_test_main(lambda: app.run(main))


class BinaryDistribution(Distribution):
  """This class is needed in order to create OS specific wheels."""

  def has_ext_modules(self):
    return False


def get_required_packages():
  """Returns list of required packages."""

  required_packages = [
      'absl-py >= 0.6.1',
      'cloudpickle >= 1.3',
      'gin-config >= 0.4.0',
      'gym >= 0.17.0, <=0.23.0',
      'numpy >= 1.13.3',
      'pillow',
      'six >= 1.10.0',
      'protobuf >= 3.11.3',
      'wrapt >= 1.11.1',
      'typing-extensions >= 3.7.4.3',
      # Used by gym >= 0.22.0. Only installed as a dependency when gym[all] is
      # installed or if gym[*] (where * is an environment which lists pygame as
      # a dependency).
      'pygame == 2.1.0',
  ]
  add_additional_packages(required_packages)
  return required_packages


def add_additional_packages(required_packages):
  """Adds additional required packages."""
  if FLAGS.release:
    tfp_version = TFP_VERSION
  else:
    tfp_version = TFP_NIGHTLY

  if FLAGS.tfp_version:
    tfp_version = FLAGS.tfp_version
  required_packages.append(tfp_version)


def get_test_packages():
  """Returns list of packages needed when testing."""
  test_packages = [
      'ale-py',
      'atari-py',  # TODO(b/200012648) contains ALE/Tetris ROM for unit test.
      'mock >= 2.0.0',
      'opencv-python >= 3.4.1.15',
      'pybullet',
      'scipy >= 1.1.0',
      'tensorflow_datasets'
  ]
  return test_packages


def get_reverb_packages():
  """Returns list of required packages if using reverb."""
  reverb_packages = []
  if FLAGS.release:
    tf_version = TENSORFLOW_VERSION
    reverb_version = REVERB_VERSION
    rlds_version = RLDS_VERSION
  else:
    tf_version = TENSORFLOW_NIGHTLY
    reverb_version = REVERB_NIGHTLY
    rlds_version = RLDS_NIGHTLY

  # Overrides required versions if FLAGS are set.
  if FLAGS.tf_version:
    tf_version = FLAGS.tf_version
  if FLAGS.reverb_version:
    reverb_version = FLAGS.reverb_version
  if FLAGS.rlds_version:
    rlds_version = FLAGS.rlds_version

  reverb_packages.append(rlds_version)
  reverb_packages.append(reverb_version)
  reverb_packages.append(tf_version)
  return reverb_packages


def get_version():
  """Returns the version and project name to associate with the build."""
  __dev_version__ = tf_agents_version.__dev_version__  # pylint: disable=invalid-name
  __rel_version__ = tf_agents_version.__rel_version__  # pylint: disable=invalid-name

  if FLAGS.release:
    version = __rel_version__
    project_name = 'tf-agents'
  else:
    version = __dev_version__
    version += datetime.datetime.now().strftime('%Y%m%d')
    project_name = 'tf-agents-nightly'
  return version, project_name


def run_setup():
  """Triggers build, install, and other features of `setuptools.setup`."""

  # Builds the long description from the README.
  root_path = os.path.abspath(os.path.dirname(__file__))
  with codecs.open(os.path.join(root_path, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

  version, project_name = get_version()
  test_packages = get_test_packages()
  setup(
      name=project_name,
      version=version,
      description='TF-Agents: A Reinforcement Learning Library for TensorFlow',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Google LLC',
      author_email='no-reply@google.com',
      url='https://github.com/tensorflow/agents',
      license='Apache 2.0',
      include_package_data=True,
      packages=find_packages(),
      install_requires=get_required_packages(),
      tests_require=test_packages,
      extras_require={
          'tests': test_packages,
          'reverb': get_reverb_packages(),
      },
      # Supports Python 3 only.
      python_requires='>=3',
      # Add in any packaged data.
      zip_safe=False,
      distclass=BinaryDistribution,
      cmdclass={
          'test': Test,
      },
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Software Development',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules',
      ],
      keywords='tensorflow agents reinforcement learning machine bandits',
  )


if __name__ == '__main__':
  # Hide argparse help so `setuptools.setup` help prints. This pattern is an
  # improvement over using `sys.argv` and then `sys.argv.remove`, which also
  # did not provide help about custom arguments.
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument(
      '--release',
      action='store_true',
      help='Pass as true to do a release build')
  parser.add_argument(
      '--tf-version',
      type=str,
      default=None,
      help='Overrides TF version required when Reverb is installed, e.g.'
      'tensorflow>=2.3.0')
  parser.add_argument(
      '--reverb-version',
      type=str,
      default=None,
      help='Overrides Reverb version required, e.g. dm-reverb>=0.1.0')
  parser.add_argument(
      '--tfp-version',
      type=str,
      default=None,
      help='Overrides tfp version required, e.g. '
      'tensorflow-probability==0.11.0rc0')
  parser.add_argument(
      '--rlds-version',
      type=str,
      default=None,
      help='Overrides rlds version required, e.g. '
      'rlds==0.1.4')
  parser.add_argument(
      '--broken_tests',
      type=str,
      default='broken_tests.txt',
      help='Broken tests file to use.')
  FLAGS, unparsed = parser.parse_known_args()
  # Go forward with only non-custom flags.
  sys.argv.clear()
  # Downstream `setuptools.setup` expects args to start at the second element.
  unparsed.insert(0, 'foo')
  sys.argv.extend(unparsed)
  run_setup()
