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

"""Tests for environments.gym_wrapper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from absl.testing.absltest import mock
import gym
import gym.spaces
import numpy as np

from tf_agents.environments import gym_wrapper
from tf_agents.utils import test_utils


class GymWrapperSpecTest(test_utils.TestCase):

  def test_spec_from_gym_space_discrete(self):
    discrete_space = gym.spaces.Discrete(3)
    spec = gym_wrapper.spec_from_gym_space(discrete_space)

    self.assertEqual((), spec.shape)
    self.assertEqual(np.int64, spec.dtype)
    self.assertEqual(0, spec.minimum)
    self.assertEqual(2, spec.maximum)

  def test_spec_from_gym_space_multi_discrete(self):
    multi_discrete_space = gym.spaces.MultiDiscrete([1, 2, 3, 4])
    spec = gym_wrapper.spec_from_gym_space(multi_discrete_space)

    self.assertEqual((4,), spec.shape)
    self.assertEqual(np.int32, spec.dtype)
    np.testing.assert_array_equal(np.array([0], dtype=np.int), spec.minimum)
    np.testing.assert_array_equal(
        np.array([0, 1, 2, 3], dtype=np.int), spec.maximum)

  def test_spec_from_gym_space_multi_binary(self):
    multi_binary_space = gym.spaces.MultiBinary(4)
    spec = gym_wrapper.spec_from_gym_space(multi_binary_space)

    self.assertEqual((4,), spec.shape)
    self.assertEqual(np.int32, spec.dtype)
    np.testing.assert_array_equal(np.array([0], dtype=np.int), spec.minimum)
    np.testing.assert_array_equal(np.array([1], dtype=np.int), spec.maximum)

  def test_spec_from_gym_space_multi_binary_2d(self):
    multi_binary_space = gym.spaces.MultiBinary((8, 8))
    spec = gym_wrapper.spec_from_gym_space(multi_binary_space)

    self.assertEqual((8, 8), spec.shape)
    self.assertEqual(np.int32, spec.dtype)
    np.testing.assert_array_equal(np.array([0], dtype=np.int), spec.minimum)
    np.testing.assert_array_equal(np.array([1], dtype=np.int), spec.maximum)

  def test_spec_from_gym_space_box_scalars(self):
    for dtype in (np.float32, np.float64):
      box_space = gym.spaces.Box(-1.0, 1.0, (3, 4), dtype=dtype)
      spec = gym_wrapper.spec_from_gym_space(box_space)

      self.assertEqual((3, 4), spec.shape)
      self.assertEqual(dtype, spec.dtype)
      np.testing.assert_array_equal(-np.ones((3, 4)), spec.minimum)
      np.testing.assert_array_equal(np.ones((3, 4)), spec.maximum)

  def test_spec_from_gym_space_box_scalars_simplify_bounds(self):
    box_space = gym.spaces.Box(-1.0, 1.0, (3, 4))
    spec = gym_wrapper.spec_from_gym_space(box_space, simplify_box_bounds=True)

    self.assertEqual((3, 4), spec.shape)
    self.assertEqual(np.float32, spec.dtype)
    np.testing.assert_array_equal(np.array([-1], dtype=np.int), spec.minimum)
    np.testing.assert_array_equal(np.array([1], dtype=np.int), spec.maximum)

  def test_spec_from_gym_space_when_simplify_box_bounds_false(self):
    # testing on gym.spaces.Dict which makes recursive calls to
    # _spec_from_gym_space
    box_space = gym.spaces.Box(-1.0, 1.0, (2,))
    dict_space = gym.spaces.Dict({'box1': box_space, 'box2': box_space})
    spec = gym_wrapper.spec_from_gym_space(
        dict_space, simplify_box_bounds=False)

    self.assertEqual((2,), spec['box1'].shape)
    self.assertEqual((2,), spec['box2'].shape)
    self.assertEqual(np.float32, spec['box1'].dtype)
    self.assertEqual(np.float32, spec['box2'].dtype)
    self.assertEqual('box1', spec['box1'].name)
    self.assertEqual('box2', spec['box2'].name)
    np.testing.assert_array_equal(np.array([-1, -1], dtype=np.int),
                                  spec['box1'].minimum)
    np.testing.assert_array_equal(np.array([1, 1], dtype=np.int),
                                  spec['box1'].maximum)
    np.testing.assert_array_equal(np.array([-1, -1], dtype=np.int),
                                  spec['box2'].minimum)
    np.testing.assert_array_equal(np.array([1, 1], dtype=np.int),
                                  spec['box2'].maximum)

  def test_spec_from_gym_space_box_array(self):
    for dtype in (np.float32, np.float64):
      box_space = gym.spaces.Box(np.array([-1.0, -2.0]), np.array([2.0, 4.0]),
                                 dtype=dtype)
      spec = gym_wrapper.spec_from_gym_space(box_space)

      self.assertEqual((2,), spec.shape)
      self.assertEqual(dtype, spec.dtype)
      np.testing.assert_array_equal(np.array([-1.0, -2.0]), spec.minimum)
      np.testing.assert_array_equal(np.array([2.0, 4.0]), spec.maximum)

  def test_spec_from_gym_space_box_array_constant_bounds(self):
    for dtype in (np.float32, np.float64):
      box_space = gym.spaces.Box(
          np.array([-1.0, -1.0]), np.array([2.0, 2.0]), dtype=dtype)
      spec = gym_wrapper.spec_from_gym_space(box_space)

      self.assertEqual((2,), spec.shape)
      self.assertEqual(dtype, spec.dtype)
      self.assertAllEqual(-1.0, spec.minimum)
      self.assertAllEqual(2.0, spec.maximum)

  def test_spec_from_gym_space_box_array_constant_min(self):
    for dtype in (np.float32, np.float64):
      box_space = gym.spaces.Box(
          np.array([-1.0, -1.0]), np.array([2.0, 4.0]), dtype=dtype)
      spec = gym_wrapper.spec_from_gym_space(box_space)

      self.assertEqual((2,), spec.shape)
      self.assertEqual(dtype, spec.dtype)
      self.assertAllEqual([-1., -1.], spec.minimum)
      self.assertAllEqual([2., 4.], spec.maximum)

  def test_spec_from_gym_space_tuple(self):
    tuple_space = gym.spaces.Tuple((gym.spaces.Discrete(2),
                                    gym.spaces.Discrete(3)))
    spec = gym_wrapper.spec_from_gym_space(tuple_space)

    self.assertEqual(2, len(spec))
    self.assertEqual((), spec[0].shape)
    self.assertEqual(np.int64, spec[0].dtype)
    self.assertEqual(0, spec[0].minimum)
    self.assertEqual(1, spec[0].maximum)

    self.assertEqual((), spec[1].shape)
    self.assertEqual(np.int64, spec[1].dtype)
    self.assertEqual(0, spec[1].minimum)
    self.assertEqual(2, spec[1].maximum)

  def test_spec_from_gym_space_tuple_mixed(self):
    tuple_space = gym.spaces.Tuple((
        gym.spaces.Discrete(2),
        gym.spaces.Box(-1.0, 1.0, (3, 4)),
        gym.spaces.Tuple((gym.spaces.Discrete(2), gym.spaces.Discrete(3))),
        gym.spaces.Dict({
            'spec_1':
                gym.spaces.Discrete(2),
            'spec_2':
                gym.spaces.Tuple((gym.spaces.Discrete(2),
                                  gym.spaces.Discrete(3))),
        }),
    ))
    spec = gym_wrapper.spec_from_gym_space(tuple_space)

    self.assertEqual(4, len(spec))
    # Test Discrete
    self.assertEqual((), spec[0].shape)
    self.assertEqual(np.int64, spec[0].dtype)
    self.assertEqual(0, spec[0].minimum)
    self.assertEqual(1, spec[0].maximum)

    # Test Box
    self.assertEqual((3, 4), spec[1].shape)
    self.assertEqual(np.float32, spec[1].dtype)
    np.testing.assert_array_almost_equal(-np.ones((3, 4)), spec[1].minimum)
    np.testing.assert_array_almost_equal(np.ones((3, 4)), spec[1].maximum)

    # Test Tuple
    self.assertEqual(2, len(spec[2]))
    self.assertEqual((), spec[2][0].shape)
    self.assertEqual(np.int64, spec[2][0].dtype)
    self.assertEqual(0, spec[2][0].minimum)
    self.assertEqual(1, spec[2][0].maximum)
    self.assertEqual((), spec[2][1].shape)
    self.assertEqual(np.int64, spec[2][1].dtype)
    self.assertEqual(0, spec[2][1].minimum)
    self.assertEqual(2, spec[2][1].maximum)

    # Test Dict
    # Test Discrete in Dict
    discrete_in_dict = spec[3]['spec_1']
    self.assertEqual((), discrete_in_dict.shape)
    self.assertEqual(np.int64, discrete_in_dict.dtype)
    self.assertEqual(0, discrete_in_dict.minimum)
    self.assertEqual(1, discrete_in_dict.maximum)

    # Test Tuple in Dict
    tuple_in_dict = spec[3]['spec_2']
    self.assertEqual(2, len(tuple_in_dict))
    self.assertEqual((), tuple_in_dict[0].shape)
    self.assertEqual(np.int64, tuple_in_dict[0].dtype)
    self.assertEqual(0, tuple_in_dict[0].minimum)
    self.assertEqual(1, tuple_in_dict[0].maximum)
    self.assertEqual((), tuple_in_dict[1].shape)
    self.assertEqual(np.int64, tuple_in_dict[1].dtype)
    self.assertEqual(0, tuple_in_dict[1].minimum)
    self.assertEqual(2, tuple_in_dict[1].maximum)

  def test_spec_from_gym_space_dict(self):
    dict_space = gym.spaces.Dict([
        ('spec_2', gym.spaces.Box(-1.0, 1.0, (3, 4))),
        ('spec_1', gym.spaces.Discrete(2)),
    ])

    spec = gym_wrapper.spec_from_gym_space(dict_space)

    keys = list(spec.keys())
    self.assertEqual('spec_1', keys[1])
    self.assertEqual(2, len(spec))
    self.assertEqual((), spec['spec_1'].shape)
    self.assertEqual(np.int64, spec['spec_1'].dtype)
    self.assertEqual(0, spec['spec_1'].minimum)
    self.assertEqual(1, spec['spec_1'].maximum)

    self.assertEqual('spec_2', keys[0])
    self.assertEqual((3, 4), spec['spec_2'].shape)
    self.assertEqual(np.float32, spec['spec_2'].dtype)
    np.testing.assert_array_almost_equal(
        -np.ones((3, 4)),
        spec['spec_2'].minimum,
    )
    np.testing.assert_array_almost_equal(
        np.ones((3, 4)),
        spec['spec_2'].maximum,
    )

  def test_spec_from_gym_space_dtype_map(self):
    class Box(gym.spaces.Box):
      """Box space without the dtype property."""

      def __init__(self, *args, **kwargs):
        super(Box, self).__init__(*args, **kwargs)
        del self.dtype

    tuple_space = gym.spaces.Tuple((
        gym.spaces.Discrete(2),
        Box(0, 1, (3, 4)),
        gym.spaces.Tuple((gym.spaces.Discrete(2), gym.spaces.Discrete(3))),
        gym.spaces.Dict({
            'spec_1':
                gym.spaces.Discrete(2),
            'spec_2':
                gym.spaces.Tuple((
                    gym.spaces.Discrete(2),
                    Box(0, 1, (3, 4)),
                )),
        }),
    ))

    dtype_map = {gym.spaces.Discrete: np.uint8, gym.spaces.Box: np.uint16}
    spec = gym_wrapper.spec_from_gym_space(tuple_space, dtype_map=dtype_map)
    self.assertEqual(np.uint8, spec[0].dtype)
    self.assertEqual(np.uint16, spec[1].dtype)
    self.assertEqual(np.uint8, spec[2][0].dtype)
    self.assertEqual(np.uint8, spec[2][1].dtype)
    self.assertEqual(np.uint8, spec[3]['spec_1'].dtype)
    self.assertEqual(np.uint8, spec[3]['spec_2'][0].dtype)
    self.assertEqual(np.uint16, spec[3]['spec_2'][1].dtype)

  def test_spec_name(self):
    box_space = gym.spaces.Box(
        np.array([-1.0, -2.0]), np.array([2.0, 4.0]), dtype=np.float32)
    spec = gym_wrapper.spec_from_gym_space(box_space, name='observation')
    self.assertEqual('observation', spec.name)

  def test_spec_name_nested(self):
    dict_space = gym.spaces.Tuple((gym.spaces.Dict({
        'spec_0':
            gym.spaces.Dict({
                'spec_1': gym.spaces.Discrete(2),
                'spec_2': gym.spaces.Discrete(2),
            }),
    }), gym.spaces.Discrete(2)))
    spec = gym_wrapper.spec_from_gym_space(dict_space, name='observation')
    self.assertEqual('observation/tuple_0/spec_0/spec_1',
                     spec[0]['spec_0']['spec_1'].name)
    self.assertEqual('observation/tuple_0/spec_0/spec_2',
                     spec[0]['spec_0']['spec_2'].name)
    self.assertEqual('observation/tuple_1', spec[1].name)


class GymWrapperOnCartpoleTest(test_utils.TestCase):

  def test_wrapped_cartpole_specs(self):
    # Note we use spec.make on gym envs to avoid getting a TimeLimit wrapper on
    # the environment.
    cartpole_env = gym.spec('CartPole-v1').make()
    env = gym_wrapper.GymWrapper(cartpole_env)

    action_spec = env.action_spec()
    self.assertEqual((), action_spec.shape)
    self.assertEqual(0, action_spec.minimum)
    self.assertEqual(1, action_spec.maximum)

    observation_spec = env.observation_spec()
    self.assertEqual((4,), observation_spec.shape)
    self.assertEqual(np.float32, observation_spec.dtype)
    high = np.array([
        4.8,
        np.finfo(np.float32).max, 2 / 15.0 * math.pi,
        np.finfo(np.float32).max
    ])
    np.testing.assert_array_almost_equal(-high, observation_spec.minimum)
    np.testing.assert_array_almost_equal(high, observation_spec.maximum)

  def test_wrapped_cartpole_reset(self):
    cartpole_env = gym.spec('CartPole-v1').make()
    env = gym_wrapper.GymWrapper(cartpole_env)

    first_time_step = env.reset()
    self.assertTrue(first_time_step.is_first())
    self.assertEqual(0.0, first_time_step.reward)
    self.assertEqual(1.0, first_time_step.discount)
    self.assertEqual((4,), first_time_step.observation.shape)
    self.assertEqual(np.float32, first_time_step.observation.dtype)

  def test_wrapped_cartpole_transition(self):
    cartpole_env = gym.spec('CartPole-v1').make()
    env = gym_wrapper.GymWrapper(cartpole_env)
    env.reset()
    transition_time_step = env.step(np.array(0, dtype=np.int32))

    self.assertTrue(transition_time_step.is_mid())
    self.assertNotEqual(None, transition_time_step.reward)
    self.assertEqual(1.0, transition_time_step.discount)
    self.assertEqual((4,), transition_time_step.observation.shape)

  def test_wrapped_cartpole_final(self):
    cartpole_env = gym.spec('CartPole-v1').make()
    env = gym_wrapper.GymWrapper(cartpole_env)
    time_step = env.reset()

    while not time_step.is_last():
      time_step = env.step(np.array(1, dtype=np.int32))

    self.assertTrue(time_step.is_last())
    self.assertNotEqual(None, time_step.reward)
    self.assertEqual(0.0, time_step.discount)
    self.assertEqual((4,), time_step.observation.shape)

  def test_get_info(self):
    cartpole_env = gym.spec('CartPole-v1').make()
    env = gym_wrapper.GymWrapper(cartpole_env)
    self.assertIsNone(env.get_info())
    env.reset()
    self.assertIsNone(None, env.get_info())
    env.step(np.array(0, dtype=np.int32))
    self.assertEqual({}, env.get_info())

  def test_automatic_reset_after_create(self):
    cartpole_env = gym.spec('CartPole-v1').make()
    env = gym_wrapper.GymWrapper(cartpole_env)

    first_time_step = env.step(0)  # pytype: disable=wrong-arg-types
    self.assertTrue(first_time_step.is_first())

  def test_automatic_reset_after_done(self):
    cartpole_env = gym.spec('CartPole-v1').make()
    env = gym_wrapper.GymWrapper(cartpole_env)
    time_step = env.reset()

    while not time_step.is_last():
      time_step = env.step(np.array(1, dtype=np.int32))

    self.assertTrue(time_step.is_last())
    first_time_step = env.step(0)  # pytype: disable=wrong-arg-types
    self.assertTrue(first_time_step.is_first())

  def test_automatic_reset_after_done_not_using_reset_directly(self):
    cartpole_env = gym.spec('CartPole-v1').make()
    env = gym_wrapper.GymWrapper(cartpole_env)
    time_step = env.step(1)  # pytype: disable=wrong-arg-types

    while not time_step.is_last():
      time_step = env.step(np.array(1, dtype=np.int32))

    self.assertTrue(time_step.is_last())
    first_time_step = env.step(0)  # pytype: disable=wrong-arg-types
    self.assertTrue(first_time_step.is_first())

  def test_method_propagation(self):
    cartpole_env = gym.spec('CartPole-v1').make()
    for method_name in ('render', 'seed', 'close'):
      setattr(cartpole_env, method_name, mock.MagicMock())
    env = gym_wrapper.GymWrapper(cartpole_env)
    env.render()
    self.assertEqual(1, cartpole_env.render.call_count)
    cartpole_env.render.assert_called_with('rgb_array')
    env.seed(0)
    self.assertEqual(1, cartpole_env.seed.call_count)
    cartpole_env.seed.assert_called_with(0)
    env.close()
    self.assertEqual(1, cartpole_env.close.call_count)

  def test_render_kwargs(self):
    cartpole_env = gym.spec('CartPole-v1').make()
    def _gym_render_mock(mode='rgb_array', **kwargs):
      del mode, kwargs  # unused
      return None
    cartpole_env.render = mock.MagicMock(side_effect=_gym_render_mock)

    # The following are hypothetical (mocked) usages of render(),
    # as the CartPole environment actually does not support kwargs for `render`.
    env = gym_wrapper.GymWrapper(
        cartpole_env, render_kwargs={'width': 96, 'height': 128})
    env.render()
    cartpole_env.render.assert_called_with('rgb_array', width=96, height=128)
    env.render(mode='human')
    cartpole_env.render.assert_called_with('human', width=96, height=128)

    # In a corner case where kwargs includes mode (got multiple values for
    # argument 'mode'), an exception will be thrown.
    cartpole_env.render.reset_mock()
    render_kwargs = dict(mode='human', width=96, height=128)
    env = gym_wrapper.GymWrapper(cartpole_env, render_kwargs=render_kwargs)
    with self.assertRaisesRegex(TypeError, 'multiple values'):
      env.render()

  def test_obs_dtype(self):
    cartpole_env = gym.spec('CartPole-v1').make()
    cartpole_env.render = mock.MagicMock()
    env = gym_wrapper.GymWrapper(cartpole_env)
    time_step = env.reset()
    self.assertEqual(env.observation_spec().dtype, time_step.observation.dtype)


if __name__ == '__main__':
  test_utils.main()
