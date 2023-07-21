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

r"""Module to reduce commands to create stable releases.

Usage:
  # Create the branch and update version file.
  python3 release_builder.py \
  --branch_hash=db00a53a865aa2760714351216da4203004f97d6 \
  --git_repo=git@github.com:tensorflow/agents.git \
  --release_number=10.0.5 \
  --working_dir=/tmp/release_build \
  --version_file=tf_agents/version.py  \
  --mode=branch

  # Create tag. Run after creating branch, building, and testing.
  python3 release_builder.py \
  --git_repo=git@github.com:tensorflow/agents.git \
  --release_number=10.0.5 \
  --working_dir=/tmp/release_build \
  --mode=tag

"""
import fileinput
import os

from absl import app
from absl import flags
from absl import logging

from git import Repo
from git.exc import GitCommandError
from git.exc import InvalidGitRepositoryError

flags.DEFINE_string('branch_hash', None, 'Hash to build release branch from.')
flags.DEFINE_string('release_number', None,
                    'Release number, e.g. 0.6.0 or 0.6.0.rc0.')
flags.DEFINE_string('version_file', None,
                    'relative path to the version file in repo')
flags.DEFINE_string('git_repo', None, 'Full github repo path.')
flags.DEFINE_string('working_dir', None,
                    'Full path to the directory to check the code out into.')
flags.DEFINE_string(
    'mode', 'branch', 'Set to "branch" to version and create branch and tag to '
    'create the tag.')

flags.mark_flag_as_required('release_number')
flags.mark_flag_as_required('git_repo')
flags.mark_flag_as_required('working_dir')
FLAGS = flags.FLAGS


class ReleaseBuilder(object):
  """Helps to build releases through scripting of steps."""

  def __init__(self, git_repo, version_file, release_number, working_dir,
               branch_hash):
    """Initialize ReleaseBuilder class.

    Args:
      git_repo: Full path to github repo.
      version_file: relative path to version file in repo.
      release_number: String representing the release number, e.g. 0.1.2 or
        0.2.3.rc1.
      working_dir: Full path to the directory to check the code out into.
      branch_hash: Git hash to use to create the new branch if needed.
    """
    self.major, self.minor, self.patch, self.release = (
        self._parse_version_input(release_number))
    self.branch_name = 'r{}.{}.{}'.format(self.major, self.minor, self.patch)
    self.tag_name = 'v{}.{}.{}'.format(self.major, self.minor, self.patch)
    self.branch_hash = branch_hash
    self.repo = self._get_repo(git_repo, working_dir)
    if version_file:
      self.version_file = os.path.join(self.repo.working_tree_dir, version_file)

  def create_release_branch(self):
    """Creates a release branch and optionally an updated version file."""
    logging.info('Create release branch %s.', self.branch_name)
    logging.info('Starting active branch:%s.', self.repo.active_branch)

    self._checkout_or_create_branch()
    if self.version_file:
      updated = self._update_version_file()
      if updated:
        self.repo.remotes.origin.push(self.branch_name)
        logging.info('Version file updated and pushed to remote %s',
                     self.repo.remotes.origin.url)

  def create_tag(self):
    """Creates a tag from the branch."""
    logging.info('Create tag %s', self.tag_name)
    logging.info('Starting active branch:%s', self.repo.active_branch)

    self._checkout_or_create_branch()
    if not any(x.name == self.tag_name for x in self.repo.tags):
      logging.info('Create %s tag.', self.tag_name)
      self.repo.create_tag(self.tag_name)
    self.repo.remotes.origin.push(self.tag_name)
    logging.info('Created tag %s', self.tag_name)

  def _parse_version_input(self, release_number):
    """Breaks release_number into major, minor, patch, and release.

    Args:
      release_number: String representing the release number, e.g. 0.1.2 or
        0.2.3.rc1.

    Returns:
      tuple:
        major: Major version number.
        minor: Minor version number.
        patch: Patch version number.
        release: Suffix for the release, e.g. rc0. Often an empty string.
    """
    parts = release_number.split('.')
    if len(parts) < 3:
      logging.error('FLAGS.version_number must be x.x.x or x.x.x.y')
    elif len(parts) == 3:
      return parts[0], parts[1], parts[2], ''
    else:
      return parts[0], parts[1], parts[2], parts[3]

  def _commit_file(self, file, msg):
    """Commits file to repo if file has changed.

    Args:
      file: Relative path to the file to commit.
      msg: Message to put in the commit.

    Returns:
      True if commit takes place or false if no change.
    """
    file_paths = [os.path.join(self.repo.working_tree_dir, file)]
    if self.repo.index.diff(None, paths=file_paths):
      index = self.repo.index
      index.add(file_paths)
      index.commit(msg)
      logging.info('%s committed locally.', file)
      return True
    else:
      return False

  def _update_version_file(self):
    """Updates the version variables in the project's version file.

    Returns:
      True if the version file was committed, False if no change was needed.
    """
    file_path = os.path.join(self.repo.working_tree_dir, self.version_file)
    self._update_version_numbers(file_path)
    return self._commit_file(
        self.version_file, 'Version updated for release {}.{}.{}{}.'.format(
            self.major, self.minor, self.patch, self.release))

  def _update_version_numbers(self, file_path):
    """Updates the version variables in the project's version file.

    This assumes that the version file using the following attributes:
    _MAJOR_VERSION, _MINOR_VERSION, _PATCH_VERSION, and _REL_SUFFIX.

    Args:
      file_path: path to the version file.
    """
    with fileinput.input(files=(file_path), inplace=True) as f:
      for line in f:
        logging.info(line)
        if line.startswith('_MAJOR_VERSION '):
          print("_MAJOR_VERSION = '{}'".format(self.major))
          continue

        if line.startswith('_MINOR_VERSION '):
          print("_MINOR_VERSION = '{}'".format(self.minor))
          continue

        if line.startswith('_PATCH_VERSION '):
          print("_PATCH_VERSION = '{}'".format(self.patch))
          continue

        if line.startswith('_REL_SUFFIX '):
          print("_REL_SUFFIX = '{}'".format(self.release))
          continue

        print(line, end='')

  def _checkout_or_create_branch(self):
    """Checkout or create branch from branch_hash provided.

    If the branch does not exist in the remote (origin) or locally then it is
    created at the provided hash point. In all cases the branch is setup with
    the upstream set to the origin.
    """
    # Checks if already on the desired branch and assumes branch was already
    # pushed to the remote.
    if self.repo.active_branch.name == self.branch_name:
      return

    if any(x.name == self.branch_name for x in self.repo.branches):
      self.repo.git.checkout(self.branch_name)
    else:
      try:
        self.repo.git.checkout('origin/' + self.branch_name, b=self.branch_name)
      except GitCommandError:
        self.repo.create_head(
            self.branch_name, commit=self.branch_hash).checkout()
        logging.info('Created branch %s from hash %s.', self.repo.active_branch,
                     self.branch_hash)

    self.repo.git.push('--set-upstream', 'origin', self.branch_name)
    logging.info('Branch pushed to remote %s.', self.repo.remotes.origin.url)
    logging.info('Active branch changed to:%s', self.repo.active_branch)

  def _get_repo(self, git_repo, working_dir):
    """Clones repo of points to existing repo.

    Args:
      git_repo: Full url to git repository.
      working_dir: Full path to the directory to check the code out into.

    Returns:
      Repo object representing the git repository.
    """
    os.makedirs(working_dir, exist_ok=True)
    try:
      return Repo(working_dir)
    except InvalidGitRepositoryError:
      return Repo.clone_from(git_repo, working_dir)


def main(_):
  logging.set_verbosity(logging.INFO)
  release_build = ReleaseBuilder(FLAGS.git_repo, FLAGS.version_file,
                                 FLAGS.release_number, FLAGS.working_dir,
                                 FLAGS.branch_hash)
  if 'branch' in FLAGS.mode:
    release_build.create_release_branch()
  elif 'tag' in FLAGS.mode:
    release_build.create_tag()
  else:
    print('Error: Unknown FLAGS.mode.')


if __name__ == '__main__':
  app.run(main)
