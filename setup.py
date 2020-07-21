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

# Default versions for packages we often override for testing and release
# candidates. These can all be overridden with flags.
TFP_VERSION = 'tensorflow-probability==0.11.0rc0'
TFP_NIGHTLY = 'tfp-nightly'
TENSORFLOW_VERSION = 'tensorflow>=2.3.0'
TENSORFLOW_NIGHTLY = 'tf-nightly'
REVERB_VERSION = 'dm-reverb'
REVERB_NIGHTLY = 'dm-reverb-nightly'


class StderrWrapper(io.IOBase):

  def write(self, *args, **kwargs):
    return sys.stderr.write(*args, **kwargs)

  def writeln(self, *args, **kwargs):
    if args or kwargs:
      sys.stderr.write(*args, **kwargs)
    sys.stderr.write('\n')


class TestLoader(unittest.TestLoader):

  def __init__(self, blacklist):
    super(TestLoader, self).__init__()
    self._blacklist = blacklist

  def _match_path(self, path, full_path, pattern):
    if not fnmatch.fnmatch(path, pattern):
      return False
    module_name = full_path.replace('/', '.').rstrip('.py')
    if any(module_name.endswith(x) for x in self._blacklist):
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

      run_separately = load_test_list('test_individually.txt')
      broken_tests = load_test_list('broken_tests.txt')

      test_loader = TestLoader(blacklist=run_separately + broken_tests)
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
      'cloudpickle == 1.3',  # TODO(b/155109696): Unpin cloudpickle version.
      'gin-config >= 0.3.0',
      'numpy >= 1.13.3',
      'six >= 1.10.0',
      'protobuf >= 3.11.3',
      'wrapt >= 1.11.1',
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
      'atari_py == 0.1.7',
      'gym == 0.12.5',
      'mock >= 2.0.0',
      'opencv-python >= 3.4.1.15',
      'pybullet',
      'scipy == 1.1.0',
  ]
  return test_packages


def get_reverb_packages():
  """Returns list of required packages if using reverb."""
  reverb_packages = []
  if FLAGS.release:
    tf_version = TENSORFLOW_VERSION
    reverb_version = REVERB_VERSION
  else:
    tf_version = TENSORFLOW_NIGHTLY
    reverb_version = REVERB_NIGHTLY

  # Overrides required versions if FLAGS are set.
  if FLAGS.tf_version:
    tf_version = FLAGS.tf_version
  if FLAGS.reverb_version:
    reverb_version = FLAGS.reverb_version

  reverb_packages.append(reverb_version)
  reverb_packages.append(tf_version)
  return reverb_packages


def get_version():
  """Returns the version and project name to associate with the build."""
  from tf_agents.version import __dev_version__  # pylint: disable=g-import-not-at-top
  from tf_agents.version import __rel_version__  # pylint: disable=g-import-not-at-top

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
      url='http://github.com/tensorflow/agents',
      license='Apache 2.0',
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
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
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
  FLAGS, unparsed = parser.parse_known_args()
  # Go forward with only non-custom flags.
  sys.argv.clear()
  # Downstream `setuptools.setup` expects args to start at the second element.
  unparsed.insert(0, 'foo')
  sys.argv.extend(unparsed)
  run_setup()
