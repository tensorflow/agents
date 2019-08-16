import functools
import time

import numpy as np
import tensorflow as tf
from tf_agents.environments import parallel_batched_py_environment, random_py_environment
from tf_agents.environments.parallel_py_environment_test import ParallelPyEnvironmentTest
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


class BatchedSlowStartingEnvironment(random_py_environment.RandomPyEnvironment):
    def __init__(self, *args, **kwargs):
        time_sleep = kwargs.pop('time_sleep', 1.0)
        time.sleep(time_sleep)
        super(BatchedSlowStartingEnvironment, self).__init__(batch_size=3, *args, **kwargs)


class ParallelBatchedPyEnvironmentTest(ParallelPyEnvironmentTest):
    def _set_default_specs(self):
        self.observation_spec = array_spec.ArraySpec((3, 3), np.float32)
        self.time_step_spec = ts.time_step_spec(self.observation_spec)
        self.action_spec = array_spec.BoundedArraySpec([7],
                                                       dtype=np.float32,
                                                       minimum=-1.0,
                                                       maximum=1.0)

    def _make_parallel_py_environment(self,
                                      constructor=None,
                                      num_envs=2,
                                      start_serially=True,
                                      blocking=True,
                                      batch_size=3):
        self._set_default_specs()
        constructor = constructor or functools.partial(
            random_py_environment.RandomPyEnvironment, self.observation_spec,
            self.action_spec, batch_size=batch_size)
        return parallel_batched_py_environment.ParallelBatchedPyEnvironment(
            env_constructors=[constructor] * num_envs, blocking=blocking,
            start_serially=start_serially)

    def test_step(self):
        num_envs = 2
        batch_size = 3
        env = self._make_parallel_py_environment(num_envs=num_envs, batch_size=batch_size)
        action_spec = env.action_spec()
        observation_spec = env.observation_spec()
        rng = np.random.RandomState()
        action = np.array([
            array_spec.sample_bounded_spec(action_spec, rng)
            for _ in range(num_envs*batch_size)
        ])
        env.reset()

        # Take one step and assert observation is batched the right way.
        time_step = env.step(action)
        self.assertEqual(num_envs*batch_size, time_step.observation.shape[0])
        self.assertAllEqual(observation_spec.shape, time_step.observation.shape[1:])
        self.assertEqual(num_envs*batch_size, action.shape[0])
        self.assertAllEqual(action_spec.shape, action.shape[1:])

        # Take another step and assert that observations have the same shape.
        time_step2 = env.step(action)
        self.assertAllEqual(time_step.observation.shape,
                            time_step2.observation.shape)
        env.close()


if __name__ == '__main__':
    tf.test.main()