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

"""Tests for tf_agents.utils.common.assert_members_are_not_overridden()."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.utils import common


class Base(object):

  def __init__(self, allowlist=(), denylist=()):
    common.assert_members_are_not_overridden(
        base_cls=Base,
        instance=self,
        allowlist=allowlist,
        denylist=denylist)

  def method1(self):
    pass

  def method2(self):
    pass


def child_class(cls, allowlist=(), denylist=()):

  class ChildNoOverrides(Base):

    def __init__(self):
      super(ChildNoOverrides, self).__init__(allowlist, denylist)

  class ChildOverrideMethod1(Base):

    def __init__(self):
      super(ChildOverrideMethod1, self).__init__(allowlist, denylist)

    def method1(self):
      return 1

  class ChildOverrideBoth(Base):

    def __init__(self):
      super(ChildOverrideBoth, self).__init__(allowlist, denylist)

    def method1(self):
      return 1

    def method2(self):
      return 1

  if cls == 'ChildNoOverrides':
    return ChildNoOverrides
  elif cls == 'ChildOverrideMethod1':
    return ChildOverrideMethod1
  elif cls == 'ChildOverrideBoth':
    return ChildOverrideBoth


class AssertMembersAreNotOverriddenTest(tf.test.TestCase):

  def testNoOverridePublic(self):
    child_cls = child_class('ChildNoOverrides')
    child_cls()

  def testValueErrorOverridePublic(self):
    child_cls = child_class('ChildOverrideMethod1')
    with self.assertRaises(ValueError):
      child_cls()

  def testAllowListedCanBeOverridden(self):
    child_cls = child_class('ChildOverrideMethod1', allowlist=('method1',))
    child_cls()

  def testNonAllowListedCannotBeOverridden(self):
    child_cls = child_class('ChildOverrideBoth', allowlist=('method1',))
    with self.assertRaises(ValueError):
      child_cls()

  def testNonDenyListedCanBeOverridden(self):
    child_cls = child_class('ChildOverrideMethod1', denylist=('method2',))
    child_cls()

  def testDenyListedCannotBeOverridden(self):
    child_cls = child_class('ChildOverrideBoth', denylist=('method2',))
    with self.assertRaises(ValueError):
      child_cls()

  def testAllowListAndDenyListRaisesError(self):
    child_cls = child_class('ChildNoOverrides',
                            allowlist=('method1',),
                            denylist=('method2',))
    with self.assertRaises(ValueError):
      child_cls()


if __name__ == '__main__':
  tf.test.main()
