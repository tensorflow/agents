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

"""Tests shifted categorical distribution."""

import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp

from tf_agents.distributions import shifted_categorical


class ShiftedCategoricalTest(tf.test.TestCase):

  def testCopy(self):
    """Confirm we can copy the distribution."""
    distribution = shifted_categorical.ShiftedCategorical(
        logits=[100.0, 100.0, 100.0], shift=2)
    copy = distribution.copy()
    with self.cached_session() as s:
      probs_np = s.run(copy.probs_parameter())
      logits_np = s.run(copy.logits_parameter())
      ref_probs_np = s.run(distribution.probs_parameter())
      ref_logits_np = s.run(distribution.logits_parameter())
    self.assertAllEqual(ref_logits_np, logits_np)
    self.assertAllEqual(ref_probs_np, probs_np)

  def testShiftedSampling(self):
    distribution = shifted_categorical.ShiftedCategorical(
        probs=[0.1, 0.8, 0.1], shift=2)
    sample = distribution.sample()
    log_prob = distribution.log_prob(sample)
    results = []

    with self.cached_session() as s:
      for _ in range(100):
        value, _ = s.run([sample, log_prob])
        results.append(value)

    results = np.array(results, dtype=np.int32)
    self.assertTrue(np.all(results >= 2))
    self.assertTrue(np.all(results <= 4))

  def testCompareToCategorical(self):
    # Use the same probabilities for normal categorical and shifted one.
    shift = 2
    probabilities = [0.3, 0.3, 0.4]
    distribution = tfp.distributions.Categorical(probs=probabilities)
    shifted_distribution = shifted_categorical.ShiftedCategorical(
        probs=probabilities, shift=shift)

    # Compare outputs of basic methods, using the same starting seed.
    tf.compat.v1.set_random_seed(1)  # required per b/131171329, only with TF2.
    sample = distribution.sample(seed=1)
    tf.compat.v1.set_random_seed(1)  # required per b/131171329, only with TF2.
    shifted_sample = shifted_distribution.sample(seed=1)

    mode = distribution.mode()
    shifted_mode = shifted_distribution.mode()

    sample, shifted_sample = self.evaluate([sample, shifted_sample])
    mode, shifted_mode = self.evaluate([mode, shifted_mode])

    self.assertEqual(shifted_sample, sample + shift)
    self.assertEqual(shifted_mode, mode + shift)

    # These functions should return the same values for shifted values.
    fns = ['cdf', 'log_cdf', 'prob', 'log_prob']
    for fn_name in fns:
      fn = getattr(distribution, fn_name)
      shifted_fn = getattr(shifted_distribution, fn_name)
      value, shifted_value = self.evaluate([fn(sample),
                                            shifted_fn(shifted_sample)])
      self.assertEqual(value, shifted_value)


if __name__ == '__main__':
  tf.test.main()
