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

"""Tests masked distributions."""
import tensorflow as tf

from tf_agents.distributions import masked


class MaskedCategoricalTest(tf.test.TestCase):

  def testCopy(self):
    """Confirm we can copy the distribution."""
    distribution = masked.MaskedCategorical([100.0, 100.0, 100.0],
                                            mask=[True, False, True])
    copy = distribution.copy()
    with self.cached_session() as s:
      probs_np = s.run(copy.probs)
      logits_np = s.run(copy.logits)
      ref_probs_np = s.run(distribution.probs)
      ref_logits_np = s.run(distribution.logits)
    self.assertAllEqual(ref_logits_np, logits_np)
    self.assertAllEqual(ref_probs_np, probs_np)

  def testMasking(self):
    distribution = masked.MaskedCategorical([100.0, 100.0, 100.0],
                                            mask=[True, False, True],
                                            neg_inf=None)
    sample = distribution.sample()
    results = []

    probs_tensor = distribution.probs
    logits_tensor = distribution.logits

    with self.cached_session() as s:
      probs_np = s.run(probs_tensor)
      logits_np = s.run(logits_tensor)

      # Draw samples & confirm we never draw a masked sample
      for _ in range(100):
        results.append(s.run(sample))

    self.assertAllEqual([0.5, 0, 0.5], probs_np)
    self.assertAllEqual([100, logits_tensor.dtype.min, 100], logits_np)
    self.assertNotIn(1, results)


if __name__ == '__main__':
  tf.test.main()
