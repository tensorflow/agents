import numpy as np
import tensorflow as tf

from tf_agents.utils import nest_utils

from tf_agents.environments.parallel_py_environment import ParallelPyEnvironment


class ParallelBatchedPyEnvironment(ParallelPyEnvironment):
    def __init__(self, env_constructors, *args, **kwargs):
        super().__init__(env_constructors, *args, **kwargs)

        assert np.all(np.array([e.batch_size for e in self._envs]) == self._envs[0].batch_size)

        self._env_batch_size = self._envs[0].batch_size
        self._batch_size = len(self._envs) * self._env_batch_size

    @property
    def batch_size(self):
        return self._batch_size

    def _stack_time_steps(self, time_steps):
        """Given a list of TimeStep, combine to one with a batch dimension."""
        if self._flatten:
            return nest_utils.fast_map_structure_flatten(
                lambda *arrays: np.concatenate(arrays) if arrays[0].ndim else np.stack(arrays),
                self._time_step_spec, *time_steps)
        else:
            return nest_utils.fast_map_structure(
                lambda *arrays: np.concatenate(arrays) if arrays[0].ndim else np.stack(arrays), *time_steps)

    def _unstack_actions(self, batched_actions):
        """Returns a list of actions from potentially nested batch of actions."""
        reshaped_actions = np.reshape(batched_actions, (-1, self._env_batch_size, *self.action_spec().shape))
        unstacked_actions = [a for a in reshaped_actions]

        return unstacked_actions
