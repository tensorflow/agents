# TF-Agents

TF-Agents is a library for Reinforcement Learning in TensorFlow.

*NOTE:* TF-Agents is under active development. Interfaces may change at any
time.


## Agents

In TF-Agents we implement learning methods under the name `Agent`. These
encompass two main responsibilities, how the model should be updated given
experience, and how a policy should be generated from the model.

The following agents are available under TF-Agents:

* DQN: __Human level control through deep reinforcement learning__ Mnih et al., 2015 https://deepmind.com/research/dqn/
* DDPG: __Continuous control with deep reinforcement learning__ Lilicrap et al.  https://arxiv.org/abs/1509.02971
* TD3: __Addressing Function Approximation Error in Actor-Critic Methods__ Fujimoto et al. https://arxiv.org/abs/1802.09477.
* REINFORCE: __Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning__ Williams http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf
* PPO: __Proximal Policy Optimization Algorithms__ Schulman et al.  http://arxiv.org/abs/1707.06347

## Tutorials

See [`tf_agents/colabs/`](https://github.com/tensorflow/tf_agents/tree/master/tf_agents/colabs/)
for tutorials on the major components provided.

## Examples
End-to-end examples training agents can be found under each agent directory.
e.g.:

* DQN: [`tf_agents/agents/dqn/examples/train_eval_gym.py`](https://github.com/tensorflow/tf_agents/tree/master/tf_agents/agents/dqn/examples/train_eval_gym.py)

## Installation

### Stable Builds

To install the latest version, run the following:

```shell
# Installing with the `--upgrade` flag ensures you'll get the latest version.
pip install --user --upgrade tfagents  # depends on TensorFlow (CPU-only)
```

TF-Agents depends on a recent stable release of
[TensorFlow](https://www.tensorflow.org/install) (pip package `tensorflow`).

Note: Since TensorFlow is *not* included as a dependency of the TF-Agents
package (in `setup.py`), you must explicitly install the TensorFlow
package (`tensorflow` or `tensorflow-gpu`). This allows us to maintain one
package instead of separate packages for CPU and GPU-enabled TensorFlow.

To force a Python 3-specific install, replace `pip` with `pip3` in the above
commands. For additional installation help, guidance installing prerequisites,
and (optionally) setting up virtual environments, see the [TensorFlow
installation guide](https://www.tensorflow.org/install).

### Nightly Builds

There are also nightly builds of TF-Agents under the pip package
`tfagents-nightly`, which requires you install on one of `tf-nightly` and
`tf-nightly-gpu`. Nightly builds include newer features, but may be less stable
than the versioned releases.

## Contributing

We're eager to collaborate with you! See [`CONTRIBUTING.md`](CONTRIBUTING.md)
for a guide on how to contribute. This project adheres to TensorFlow's
[code of conduct](CODE_OF_CONDUCT.md). By participating, you are expected to
uphold this code.

## References

* # TODO(oars): How do we want to be referenced?
