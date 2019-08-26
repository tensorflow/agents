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

"""Install tf_agents."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import datetime
import fnmatch
import io
import os
import subprocess
import sys
import unittest

import concurrent.futures
from setuptools import find_packages
from setuptools import setup
from setuptools.command.test import test as TestCommandBase
from setuptools.dist import Distribution


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


class Test(TestCommandBase):

  def run_tests(self):
    # Import absl inside run, where dependencies have been loaded already.
    from absl import app  # pylint: disable=g-import-not-at-top

    def main(_):
      # pybullet imports multiprocessing in their setup.py, which causes an
      # issue when we import multiprocessing.pool.dummy down the line because
      # the PYTHONPATH has changed.
      for module in ['multiprocessing', 'multiprocessing.pool',
                     'multiprocessing.dummy', 'multiprocessing.pool.dummy']:
        if module in sys.modules:
          del sys.modules[module]
      # Reimport multiprocessing to avoid spurious error printouts. See
      # https://bugs.python.org/issue15881.
      import multiprocessing as _  # pylint: disable=g-import-not-at-top
      integration_tests = [
          x.rstrip()
          for x in open('integration_tests.txt', 'r').readlines()
          if x
      ]
      # Remove comments and blanks after comments are removed.
      integration_tests = [
          x.partition('#')[0].strip() for x in integration_tests
      ]
      integration_tests = [x.split(' ') for x in integration_tests if x]

      run_separately = [
          x.rstrip() for x in open('test_individually.txt', 'r').readlines()
          if x]
      # Remove comments and blanks after comments are removed.
      run_separately = [x.partition('#')[0].strip() for x in run_separately]
      run_separately = [x for x in run_separately if x]
      test_loader = TestLoader(blacklist=run_separately)
      test_suite = test_loader.discover('tf_agents', pattern='*_test.py')
      stderr = StderrWrapper()
      result = unittest.TextTestResult(stderr, descriptions=True, verbosity=2)
      test_suite.run(result)
      external_test_failures = []
      def _exec(args):
        try:
          return subprocess.check_output(args, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
          stderr.writeln(str(e.output))
          external_test_failures.append(e)

      external_tests = []
      for test in run_separately:
        external_tests.append('tf_agents/{}.py'.format(test.replace('.', '/')))
      external_tests = external_tests + integration_tests

      # Integration tests require a unique --root_dir argument set.
      temp_dir_index = 0
      with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for filename in external_tests:
          if isinstance(filename, list):  # Integration test.
            args = [sys.executable] + filename + [
                '--root_dir=/tmp/test_{}_{}'.format(os.getpid(), temp_dir_index)
            ]
            temp_dir_index += 1
          else:
            args = [sys.executable, filename]
          futures.append(executor.submit(_exec, args))
        for future in concurrent.futures.as_completed(futures):
          print(future.result())

      for failure in external_test_failures:
        stderr.writeln(str(failure))

      final_output = ('Tests run: {} grouped and {} external.  '.format(
          result.testsRun, len(external_tests)) +
                      'Errors: {}  Failures: {}  External failures: {}.'.format(
                          len(result.errors), len(result.failures),
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
    return app.run(main)


from tf_agents.version import __dev_version__  # pylint: disable=g-import-not-at-top
from tf_agents.version import __rel_version__  # pylint: disable=g-import-not-at-top

REQUIRED_PACKAGES = [
    'absl-py >= 0.6.1',
    'gin-config == 0.1.3',
    'numpy >= 1.13.3',
    'six >= 1.10.0',
    # tensorflow-probability added below
]


TEST_REQUIRED_PACKAGES = [
    'atari_py == 0.1.7',
    'gym == 0.12.5',
    'opencv-python >= 3.4.1.15',
    'pybullet',
    'scipy == 1.1.0',
]

REQUIRED_TFP_VERSION = '0.6.0'

if '--release' in sys.argv:
  release = True
  sys.argv.remove('--release')
  version = __rel_version__
else:
  # Build a nightly package by default.
  release = False
  version = __dev_version__
  version += datetime.datetime.now().strftime('%Y%m%d')

if release:
  project_name = 'tf-agents'
  tfp_package_name = 'tensorflow-probability>={}'.format(REQUIRED_TFP_VERSION)
else:
  # Nightly releases use date-based versioning of the form
  # '0.0.1.dev20180305'
  project_name = 'tf-agents-nightly'
  tfp_package_name = 'tfp-nightly'

REQUIRED_PACKAGES.append(tfp_package_name)

if sys.version_info.major == 2:
  # mock comes with unittest.mock for python3, need to install for
  # python2
  REQUIRED_PACKAGES.append('mock >= 2.0.0')


class BinaryDistribution(Distribution):
  """This class is needed in order to create OS specific wheels."""

  def has_ext_modules(self):
    return False

here = os.path.abspath(os.path.dirname(__file__))
with codecs.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()

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
    install_requires=REQUIRED_PACKAGES,
    tests_require=TEST_REQUIRED_PACKAGES,
    extras_require={'tests': TEST_REQUIRED_PACKAGES},
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
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='tensorflow agents reinforcement learning machine learning',
)
