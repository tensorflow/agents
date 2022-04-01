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

"""Actions sampler that supports sampling the following types of actions.

1. The structure of the action must be a dict, not a nested structure.
2. It contains one (and only one) categorical (one_hot) action.
   The category action should define a spec whose spec.dtype.is_integer==True.
   The size of this category action indicate the total number of mutually
   exclusive actions.
3. Each category is 0/1 action by default. But it can also extends to stand for
   a continuous action. In order to do that, sub_actions_fields must be set.
4. The sampler will always sample 0/1 actions 1 for each categorical and sample
   continuous actions evenly (for each category) among the rest of samples.

For example,

```
action_spec = {
    'continuous1': tensor_spec.BoundedTensorSpec(
        [2], tf.float32, 0.0, 1.0),
    'continuous2': tensor_spec.BoundedTensorSpec(
        [2], tf.float32, 0.0, 1.0),
    'continuous3': tensor_spec.BoundedTensorSpec(
        [2], tf.float32, 0.0, 1.0),
    'categorical': tensor_spec.BoundedTensorSpec(
        [4], tf.int32, 0, 1)}

sampler = (
    qtopt_cem_actions_sampler_continuous_and_one_hot.GaussianActionsSampler(
        action_spec=action_spec, sample_clippers=[[], [], []],
        sub_actions_fields=[
            ['categorical'], ['continuous1', 'continuous3'], ['continuous2']]))
```

In this case, the action has a 'categorical' field that is a categorical
action. There are 4 possible actions:
[1, 0, 0, 0] -- choosing 1st 0/1 action
[0, 1, 0, 0] -- choosing 2nd 0/1 action
[0, 0, 1, 0] -- choosing continuous1 & continuous3
[0, 0, 0, 1] -- choosing continuous2

when 1st or 2nd action is chosen, all continuous fields will be all 0s
when 3rd action is chosen, 'continuous2' will be all 0s
when 4th action is chosen, 'continuous1', 'continuous3' will be all 0s

Therefore making the 4 actions mutually exclusive.

For example, if batch_size == 1 and you call
```
sampler.sample_batch_and_clip(4)
```
The result will be

{
'continuous1':
  [[[0.0, 0.0],
    [0.0, 0.0],
    [0.3, 0.5],
    [0.0, 0.0]]],
'continuous2':
  [[[0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.2, 0.1]]],
'continuous3':
  [[[0.0, 0.0],
    [0.0, 0.0],
    [0.4, 0.6],
    [0.0, 0.0]]],
'categorical':
  [[[1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]]]
}


Changing the order of the sub_action_fields will change the what each
one_hot_vector stands for.

For example, if you call
```
sampler = (
    qtopt_cem_actions_sampler_continuous_and_one_hot.GaussianActionsSampler(
        action_spec=action_spec, sample_clippers=[[], [], []],
        sub_actions_fields=[
            ['continuous1', 'continuous3'], ['categorical'], ['continuous2']]))
```

[1, 0, 0, 0] -- choosing continuous1 & continuous3
[0, 1, 0, 0] -- choosing 1st categorical action
[0, 0, 1, 0] -- choosing 2nd categorical action
[0, 0, 0, 1] -- choosing continuous2

Some notations used in the comments below are:

B: batch_size
A: action_size
N: num_samples
S: num_mutually_exclusive_actions
K: num_sub_continuous_actions
S-K: num_sub_categorical_actions

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.policies.samplers import qtopt_cem_actions_sampler
from tf_agents.utils import common


@gin.configurable
class GaussianActionsSampler(qtopt_cem_actions_sampler.ActionsSampler):
  """Action sampler that samples continuous actions using Gaussian distribution and one_hot actions.

  Supports dict action_spec with arbitrary 1d continuous actions and 1 one_hot
  action.

  Given a batch of distribution params(including mean and var) for K (K >= 1)
  mutually exclusive continuous actions, sample [B, N-S+K, A] continuous actions
  and [B, S-K, S] one_hot actions,  where 'N' means num_samples,
  'B' means batch_size, 'A' means action_size_continuous,
  'S' means num_mutually_exclusive_actions. S-K is the possible number of
  categorical actions. If K > 1, the number of samples: N-S+K will be divided
  evenly among these mutually exclusive continuous actions.
  """

  def __init__(self, action_spec, sample_clippers=None,
               sub_actions_fields=None, sample_rejecters=None,
               max_rejection_iterations=10):
    """Builds a GaussianActionsSampler.

    Args:
      action_spec: A dict of BoundedTensorSpec representing the actions.
      sample_clippers: A list of list of sample clipper functions. The function
        takes a dict of Tensors of actions and a dict of Tensors of the state,
        output a dict of Tensors of clipped actions.
      sub_actions_fields: A list of list of action keys to group
        fields into sub_actions.
      sample_rejecters: A list of callables that will reject samples and return
        a mask tensor.
      max_rejection_iterations: max_rejection_iterations
    """

    super(GaussianActionsSampler, self).__init__(
        action_spec, sample_clippers, sample_rejecters)

    num_one_hot_action = 0
    for flat_action_spec in tf.nest.flatten(action_spec):
      if flat_action_spec.shape.rank != 1:
        raise ValueError('Only 1d action is supported by this sampler. '
                         'The action_spec: \n{}\n contains action whose rank is'
                         ' not 1. Consider coverting it into multiple 1d '
                         'actions.'.format(action_spec))
      if flat_action_spec.dtype.is_integer:
        num_one_hot_action = num_one_hot_action + 1
        # S
        self._num_mutually_exclusive_actions = (
            flat_action_spec.shape.as_list()[0])

    if num_one_hot_action != 1:
      raise ValueError('Only continuous action + 1 one_hot action is supported'
                       ' by this sampler. The action_spec: \n{}\n contains '
                       'either multiple one_hot actions or no one_hot '
                       'action'.format(action_spec))

    if sample_clippers is None:
      raise ValueError('Sampler clippers must be set!')

    if sub_actions_fields is None:
      raise ValueError('sub_actions_fields must be set!')

    if len(sample_clippers) != len(sub_actions_fields):
      raise ValueError('Number of sample_clippers must be the same as number of'
                       ' sub_actions_fields! sample_clippers: {}, '
                       'sub_actions_fields: {}'.format(
                           sample_clippers, sub_actions_fields))

    if self._sample_rejecters is None:
      self._sample_rejecters = [None] * len(sub_actions_fields)

    self._max_rejection_iterations = tf.constant(max_rejection_iterations)

    self._num_sub_actions = len(sample_clippers)
    self._sub_actions_fields = sub_actions_fields

    action_spec_keys = list(sorted(self._action_spec.keys()))
    sub_actions_fields_keys = [
        item for sublist in self._sub_actions_fields for item in sublist  # pylint: disable=g-complex-comprehension
    ]
    sub_actions_fields_keys.sort()
    if action_spec_keys != sub_actions_fields_keys:
      raise ValueError('sub_actions_fields must cover all keys in action_spec!'
                       'action_spec_keys: {}, sub_actions_fields_keys:'
                       ' {}'.format(action_spec_keys, sub_actions_fields_keys))

    self._categorical_index = -1
    for i in range(self._num_sub_actions):
      if (len(self._sub_actions_fields[i]) == 1 and
          self._action_spec[self._sub_actions_fields[i][0]].dtype.is_integer):
        self._categorical_index = i
        break

    if self._categorical_index == -1:
      raise ValueError('Categorical action cannot be grouped together w/ '
                       'continuous action into a sub_action.')
    self._categorical_key = self._sub_actions_fields[self._categorical_index][0]

    # K
    self._num_sub_continuous_actions = self._num_sub_actions - 1
    # S-K
    self._num_sub_categorical_actions = (
        self._num_mutually_exclusive_actions -
        self._num_sub_continuous_actions)

    # Because the sampler will sample for all fields and there are actions
    # that are mutually exclusive. Therefore masks are needed to zero
    # out the fields that does not belong to the sub_action.
    self._masks = []
    for i in range(self._num_sub_actions):
      mask = {}
      for k in self._action_spec.keys():
        if k in self._sub_actions_fields[i]:
          mask[k] = tf.ones([1])
        else:
          mask[k] = tf.zeros([1])
      self._masks.append(mask)

    self._index_range_min = {}
    self._index_range_max = {}

  def refit_distribution_to(self, target_sample_indices, samples):
    """Refits distribution according to actions with index of ind.

    Args:
      target_sample_indices: A [B, M] sized tensor indicating the index
      samples: A dict corresponding to action_spec. Each action is
        a [B, N, A] sized tensor.

    Returns:
      mean: A dict containing [B, A] sized tensors where each row
        is the refitted mean.
      var: A dict containing [B, A] sized tensors where each row
        is the refitted var.
    """

    def get_mean(best_samples, spec, index_range_min, index_range_max):
      if spec.dtype.is_integer:
        return tf.zeros([tf.shape(target_sample_indices)[0], spec.shape[0]])
      else:
        # In the following we use a customized way to calculate mean and var
        # from best_samples. The reason why we don't use standard tf.nn.moment
        # is because:
        # 1. We only want to calculate mean and var for continuous samples
        # 2. M elites may contain both continuous and categorical samples
        # 3. In a batch (B) of data, numner of continuous elite samples may be
        #    different
        # Also because the value of samples of categorical actions are all 0.0
        # We calculate mean and var in the following way.
        # mean = sum_elites_continuou / num_elites_continuous_expanded
        # var = sum((best_samples - mean)^2 - (mean)^2 * num_elites_categorical)
        # / num_elites_continuous

        sum_elites_continuous = tf.reduce_sum(best_samples, axis=1)  # [B, A]

        # num_elites_continuous: [B]
        num_elites_continuous = tf.reduce_sum(tf.cast(
            tf.logical_and(tf.greater_equal(
                target_sample_indices, index_range_min), tf.less(
                    target_sample_indices, index_range_max)),
            tf.float32), axis=1)

        # num_elites_continuous_expanded: [B, A]
        num_elites_continuous_expanded = tf.tile(tf.expand_dims(
            num_elites_continuous, 1), [1, spec.shape.as_list()[0]])

        # mean: [B, A]
        mean = tf.math.divide_no_nan(
            sum_elites_continuous, num_elites_continuous_expanded)

        return mean

    def get_var(best_samples, mean, spec, index_range_min, index_range_max):
      if spec.dtype.is_integer:
        return tf.zeros([tf.shape(target_sample_indices)[0], spec.shape[0]])
      else:
        # num_elites_continuous: [B]
        num_elites_continuous = tf.reduce_sum(tf.cast(
            tf.logical_and(tf.greater_equal(
                target_sample_indices, index_range_min), tf.less(
                    target_sample_indices, index_range_max)),
            tf.float32), axis=1)

        # num_elites_continuous_expanded: [B, A]
        num_elites_continuous_expanded = tf.tile(tf.expand_dims(
            num_elites_continuous, 1), [1, spec.shape.as_list()[0]])

        num_elites = tf.cast(tf.shape(target_sample_indices)[1], tf.float32)
        # num_elites_categorical_expanded: [B, A]
        num_elites_categorical_expanded = (num_elites -
                                           num_elites_continuous_expanded)

        # mean_expanded: [M, B, A]
        mean_expanded = mean * tf.ones(
            [tf.shape(target_sample_indices)[1], 1, 1])
        # mean_expanded: [B, M, A]
        mean_expanded = tf.transpose(mean_expanded, [1, 0, 2])

        var = tf.math.divide_no_nan(
            tf.reduce_sum(tf.square(best_samples - mean_expanded), axis=1) -
            tf.multiply(tf.square(mean), num_elites_categorical_expanded),
            num_elites_continuous_expanded)
        return var

    best_samples = tf.nest.map_structure(
        lambda s: tf.gather(s, target_sample_indices, batch_dims=1), samples)

    if not self._index_range_min or not self._index_range_max:
      raise ValueError('sample_batch_and_clip must be called before '
                       'refit_distribution_to!')

    mean = tf.nest.map_structure(
        get_mean, best_samples, self._action_spec,
        self._index_range_min, self._index_range_max)
    var = tf.nest.map_structure(
        get_var, best_samples, mean, self._action_spec,
        self._index_range_min, self._index_range_max)

    return mean, var

  def _sample_continuous_and_transpose(
      self, mean, var, state, i, one_hot_index):
    num_samples = self._number_samples_all[i]

    def sample_and_transpose(mean, var, spec, mask):
      if spec.dtype.is_integer:
        sample = tf.one_hot(
            one_hot_index, self._num_mutually_exclusive_actions)
        sample = tf.broadcast_to(
            sample,
            [tf.shape(mean)[0],
             tf.constant(num_samples),  # pylint: disable=cell-var-from-loop
             tf.shape(mean)[1]])
      else:
        dist = tfp.distributions.Normal(loc=mean, scale=tf.sqrt(var))
        # Transpose to [B, N, A]
        sample = tf.transpose(
            dist.sample(num_samples), [1, 0, 2])  # pylint: disable=cell-var-from-loop
        sample = sample * mask
      return tf.cast(sample, spec.dtype)

    batch_size = tf.shape(tf.nest.flatten(mean)[0])[0]

    def sample_fn(mean_sample, var_sample, state_sample):
      # [B, N, A]
      samples_continuous = tf.nest.map_structure(sample_and_transpose,
                                                 mean_sample, var_sample,
                                                 self._action_spec,
                                                 self._masks[i])

      if self._sample_clippers[i]:
        for sample_clipper in self._sample_clippers[i]:
          samples_continuous = sample_clipper(samples_continuous, state_sample)

      samples_continuous = tf.nest.map_structure(
          common.clip_to_spec, samples_continuous, self._action_spec)
      return samples_continuous

    @tf.function
    def rejection_sampling(sample_rejector):
      valid_batch_samples = tf.nest.map_structure(
          lambda spec: tf.TensorArray(spec.dtype, size=batch_size),
          self._action_spec)

      for b_indx in tf.range(batch_size):
        k = tf.constant(0)
        # pylint: disable=cell-var-from-loop
        valid_samples = tf.nest.map_structure(
            lambda spec: tf.TensorArray(spec.dtype, size=num_samples),
            self._action_spec)

        count = tf.constant(0)
        while count < self._max_rejection_iterations:
          count += 1
          mean_sample = tf.nest.map_structure(
              lambda t: tf.expand_dims(tf.gather(t, b_indx), axis=0), mean)
          var_sample = tf.nest.map_structure(
              lambda t: tf.expand_dims(tf.gather(t, b_indx), axis=0), var)
          if state is not None:
            state_sample = tf.nest.map_structure(
                lambda t: tf.expand_dims(tf.gather(t, b_indx), axis=0), state)
          else:
            state_sample = None

          samples = sample_fn(mean_sample, var_sample, state_sample)  # n, a

          mask = sample_rejector(samples, state_sample)

          mask = mask[0, ...]
          mask_index = tf.where(mask)[:, 0]

          num_mask = tf.shape(mask_index)[0]
          if num_mask == 0:
            continue

          good_samples = tf.nest.map_structure(
              lambda t: tf.gather(t, mask_index, axis=1)[0, ...], samples)

          for sample_idx in range(num_mask):
            if k >= num_samples:
              break
            valid_samples = tf.nest.map_structure(
                lambda gs, vs: vs.write(k, gs[sample_idx:sample_idx+1, ...]),
                good_samples, valid_samples)
            k += 1

        if k < num_samples:
          def sample_zero_and_one_hot(spec):
            if spec.dtype.is_integer:
              sample = tf.one_hot(
                  one_hot_index, self._num_mutually_exclusive_actions)
            else:
              sample = tf.zeros(spec.shape, spec.dtype)

            sample = tf.broadcast_to(
                sample,
                tf.TensorShape([num_samples] + sample.shape.dims))
            return tf.cast(sample, spec.dtype)

          zero_samples = tf.nest.map_structure(
              sample_zero_and_one_hot, self._action_spec)
          for sample_idx in range(num_samples-k):
            valid_samples = tf.nest.map_structure(
                lambda gs, vs: vs.write(k, gs[sample_idx:sample_idx+1, ...]),
                zero_samples, valid_samples)

        valid_samples = tf.nest.map_structure(lambda vs: vs.concat(),
                                              valid_samples)

        valid_batch_samples = tf.nest.map_structure(
            lambda vbs, vs: vbs.write(b_indx, vs), valid_batch_samples,
            valid_samples)

      samples_continuous = tf.nest.map_structure(
          lambda a: a.stack(), valid_batch_samples)
      return samples_continuous

    if self._sample_rejecters[i]:
      samples_continuous = rejection_sampling(self._sample_rejecters[i])
      def set_b_n_shape(t):
        t.set_shape(tf.TensorShape([None, num_samples] + t.shape[2:].dims))

      tf.nest.map_structure(set_b_n_shape, samples_continuous)
    else:
      samples_continuous = sample_fn(mean, var, state)

    return samples_continuous

  @gin.configurable
  def sample_batch_and_clip(self, num_samples, mean, var, state=None):
    """Samples and clips a batch of actions [B, N, A] with mean and var.

    Args:
      num_samples: Number of actions to sample each round.
      mean: A dict containing [B, A] shaped tensor representing the
        mean of the actions to be sampled.
      var: A dict containing [B, A] shaped tensor representing the
        variance of the actions to be sampled.
      state: A dict of state tensors constructed according to oberservation_spec
        of the task.

    Returns:
      actions:  A dict containing tensor of sampled actions with
        shape [B, N, A]
    """
    # At least one sample for each kind of one hot action is generated.
    assert num_samples >= self._num_mutually_exclusive_actions
    num_samples_continuous_total = (
        num_samples -
        self._num_mutually_exclusive_actions +
        self._num_sub_actions - 1)
    num_samples_continuous_each = (
        num_samples_continuous_total // self._num_sub_continuous_actions)

    # When sampling N samples, we use min_index and max_index to cut N samples
    # into several segments for each sub_actions.
    min_index = 0
    max_index = 0
    self._number_samples_all = []
    for i in range(self._num_sub_actions):
      min_index = max_index
      if i == self._categorical_index:
        max_index += self._num_sub_categorical_actions
      elif i == self._num_sub_actions - 1:
        max_index = num_samples
      else:
        max_index += num_samples_continuous_each

      for k in self._action_spec.keys():
        if k in self._sub_actions_fields[i]:
          self._index_range_min[k] = min_index
          self._index_range_max[k] = max_index

      self._number_samples_all.append(max_index - min_index)

    samples_all = []

    one_hot_index = 0
    for i in range(self._num_sub_actions):
      if i == self._categorical_index:
        # Samples one_hot actions.
        def sample_one_hot(mean, spec):
          if spec.dtype.is_integer:
            full_one_hot = tf.eye(
                self._num_mutually_exclusive_actions,
                dtype=tf.int32)  # [S, S]

            categorical_one_hot = tf.gather(
                full_one_hot,
                tf.range(one_hot_index,
                         one_hot_index+self._num_sub_categorical_actions))

            return tf.broadcast_to(
                categorical_one_hot,
                [tf.shape(mean)[0],
                 self._num_sub_categorical_actions,
                 spec.shape[0]])
          else:
            return tf.zeros([
                tf.shape(mean)[0],
                self._num_sub_categorical_actions,
                spec.shape[0]])

        samples_one_hot = tf.nest.map_structure(
            sample_one_hot, mean, self._action_spec)

        samples_one_hot = tf.nest.map_structure(
            common.clip_to_spec, samples_one_hot, self._action_spec)

        samples_all.append(samples_one_hot)

        one_hot_index += self._num_sub_categorical_actions
      else:
        samples_continuous = self._sample_continuous_and_transpose(
            mean, var, state, i, one_hot_index)
        samples_all.append(samples_continuous)

        one_hot_index += 1

    samples_all = tf.nest.map_structure(
        lambda *tensors: tf.concat(tensors, axis=1),
        *samples_all)

    return samples_all
