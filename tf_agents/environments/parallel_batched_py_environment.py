import numpy as np
import tensorflow as tf

from tf_agents.utils import nest_utils

from tf_agents.environments.parallel_py_environment import ParallelPyEnvironment


class ParallelBatchedPyEnvironment(ParallelPyEnvironment):
    @property
    def batch_size(self):
        return sum(e.batch_size for e in self._envs)

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
        flattened_actions = tf.nest.flatten(batched_actions)
        if self._flatten:
            unstacked_actions = zip(*flattened_actions)
        else:
            unstacked_actions = [
                tf.nest.pack_sequence_as(batched_actions, actions)
                for actions in zip(*flattened_actions)
            ]
        return unstacked_actions
