# coding=utf-8
# Copyright 2018 The TFAgents Authors.
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
import datetime
import fnmatch
import io
import os
import subprocess
import sys
import unittest

from setuptools import find_packages  # pylint: disable=g-import-not-at-top
from setuptools import setup
from setuptools.command.install import install as InstallCommandBase
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

  def run(self):
    # Import absl inside run, where dependencies have been loaded already.
    from absl import app  # pylint: disable=g-import-not-at-top

    def run_tests(_):
      run_separately = [
          x.rstrip() for x in open('test_individually.txt', 'r').readlines()
          if x]
      test_loader = TestLoader(blacklist=run_separately)
      test_suite = test_loader.discover('tf_agents', pattern='*_test.py')
      stderr = StderrWrapper()
      result = unittest.TextTestResult(stderr, descriptions=True, verbosity=2)
      test_suite.run(result)

      external_test_failures = []

      subprocess.call([sys.executable, __file__, 'develop'])
      for test in run_separately:
        filename = 'tf_agents/%s.py' % test.replace('.', '/')
        try:
          subprocess.check_call([sys.executable, filename])
        except subprocess.CalledProcessError as e:
          external_test_failures.append(e)
      subprocess.call([sys.executable, __file__, 'develop', '--uninstall'])

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
    return app.run(run_tests)


# To enable importing version.py directly, we add its path to sys.path.
version_path = os.path.join(
    os.path.dirname(__file__), 'tf_agents', 'python')
sys.path.append(version_path)
from version import __version__  # pylint: disable=g-import-not-at-top

REQUIRED_PACKAGES = [
    'six >= 1.10.0',
    'numpy >= 1.13.3',
    # TODO(ebrevdo): Put a version number here once gin is on pypi.
    'gin-config',
]


TEST_REQUIRED_PACKAGES = [
    # Do not include baselines; it has a dependency on non-nightly TF which
    # breaks everything else.
    #    'baselines >= 0.1.5',
    'gym >= 0.10.8',
    'pybullet >= 2.3.2',
    'atari_py >= 0.1.6',
]

REQUIRED_TENSORFLOW_VERSION = '1.10.0'

if '--release' in sys.argv:
  release = True
  sys.argv.remove('--release')
else:
  # Build a nightly package by default.
  release = False

if release:
  project_name = 'tf-agents'
  tfp_package_name = 'tensorflow-probability>={}'.format(
      REQUIRED_TENSORFLOW_VERSION)
else:
  # Nightly releases use date-based versioning of the form
  # '0.0.1.dev20180305'
  project_name = 'tf-agents-nightly'
  datestring = datetime.datetime.now().strftime('%Y%m%d')
  __version__ += datestring
  tfp_package_name = 'tfp-nightly'

REQUIRED_PACKAGES.append(tfp_package_name)


class BinaryDistribution(Distribution):
  """This class is needed in order to create OS specific wheels."""

  def has_ext_modules(self):
    return False


setup(
    name=project_name,
    version=__version__,
    description='Reinforcement Learning in TensorFlow',
    author='Google LLC',
    author_email='no-reply@google.com',
    url='http://github.com/tensorflow/agents',
    license='Apache 2.0',
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    tests_require=TEST_REQUIRED_PACKAGES + REQUIRED_PACKAGES,
    # Add in any packaged data.
    zip_safe=False,
    distclass=BinaryDistribution,
    cmdclass={
        'pip_pkg': InstallCommandBase,
        'test': Test,
    },
    classifiers=[
        'Development Status :: 4 - Beta',
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
