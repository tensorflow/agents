# TF-Agents: A reliable, scalable and easy to use Reinforcement Learning library for TensorFlow.


[TF-Agents](https://github.com/tensorflow/agents) makes designing, implementing
and testing new RL algorithms easier, by providing well tested modular
components that can be modified and extended. It enables fast code iteration,
with good test integration and benchmarking.

To get started, we recommend checking out one of our Colab tutorials. If you
need an intro to RL (or a quick recap),
[start here](tf_agents/colabs/0_intro_rl.ipynb). Otherwise, check out our
[DQN tutorial](tf_agents/colabs/1_dqn_tutorial.ipynb) to get an agent up and
running in the Cartpole environment.

*RELEASE:* New 0.3.0rc0 release available.

*NOTE:* Current TF-Agents pre-release is under active development and
interfaces may change at any time. Feel free to provide feedback and comments.

## Table of contents

<a href='#Agents'>Agents</a><br>
<a href='#Tutorials'>Tutorials</a><br>
<a href='#Multi-Armed Bandits'>Multi-Armed Bandits</a><br>
<a href='#Examples'>Examples</a><br>
<a href='#Installation'>Installation</a><br>
<a href='#Contributing'>Contributing</a><br>
<a href='#Releases'>Releases</a><br>
<a href='#Principles'>Principles</a><br>
<a href='#Citation'>Citation</a><br>
<a href='#Disclaimer'>Disclaimer</a><br>

<a id='Agents'></a>
## Agents


In TF-Agents, the core elements of RL algorithms are implemented as `Agents`.
An agent encompasses two main responsibilities: defining a Policy to interact
with the Environment, and how to learn/train that Policy from collected
experience.

Currently the following algorithms are available under TF-Agents:

* [DQN: __Human level control through deep reinforcement learning__ Mnih et al., 2015](https://deepmind.com/research/dqn/)
* [DDQN: __Deep Reinforcement Learning with Double Q-learning__ Hasselt et al., 2015](https://arxiv.org/abs/1509.06461)
* [DDPG: __Continuous control with deep reinforcement learning__ Lillicrap et al., 2015](https://arxiv.org/abs/1509.02971)
* [TD3: __Addressing Function Approximation Error in Actor-Critic Methods__ Fujimoto et al., 2018](https://arxiv.org/abs/1802.09477)
* [REINFORCE: __Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning__ Williams, 1992](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf)
* [PPO: __Proximal Policy Optimization Algorithms__ Schulman et al., 2017](https://arxiv.org/abs/1707.06347)
* [SAC: __Soft Actor Critic__ Haarnoja et al., 2018](https://arxiv.org/abs/1812.05905)

<a id='Tutorials'></a>
## Tutorials

See [`tf_agents/colabs/`](tf_agents/colabs/) for tutorials on the major
components provided.

<a id='Multi-Armed Bandits'></a>

## Multi-Armed Bandits

The TF-Agents library contains also a Multi-Armed Bandits suite with a few
environments and agents. RL agents can also be used on Bandit environments. For
a tutorial, see
[`tf_agents/bandits/colabs/bandits_tutorial.ipynb`](https://github.com/tensorflow/agents/tree/master/tf_agents/bandits/colabs/bandits_tutorial.ipynb).
For examples ready to run, see
[`tf_agents/bandits/agents/examples/`](https://github.com/tensorflow/agents/tree/master/tf_agents/bandits/agents/examples/).

<a id='Examples'></a>
## Examples
End-to-end examples training agents can be found under each agent directory.
e.g.:

* DQN: [`tf_agents/agents/dqn/examples/v1/train_eval_gym.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/agents/dqn/examples/v1/train_eval_gym.py)

<a id='Installation'></a>
## Installation

TF-Agents publishes nightly and stable builds. For a list of releases read the
<a href='#Releases'>Releases</a> section. The commands below cover installing
TF-Agents stable and nightly from [pypi.org](https://pypi.org) as well as from a
GitHub clone.

### Stable

Run the commands below to install the most recent stable release (0.3.0), which
was tested with TensorFlow 1.15.0 and 2.0.0.

```bash
pip install --user tf-agents
pip install --user tensorflow==2.0.0

# Or For TensorFlow 1.x
pip install --user tensorflow==1.15.0

# To get the matching examples and colabs
git clone https://github.com/tensorflow/agents.git
cd agents
git checkout v0.3.0

```

Note: TF-Agents 0.3.0 is not compatible with TensorFlow 2.1.0 unless the nightly
release of `TensorFlow Probability` is installed: `pip install tfp-nightly`

### Nightly

Nightly builds include newer features, but may be less stable than the versioned
releases. The nightly build is pushed as `tf-agents-nightly`. We suggest
installing nightly versions of TensorFlow (`tf-nightly`) and TensorFlow
Probability (`tfp-nightly`) as those are the version TF-Agents nightly are
tested against.

To install the nightly build version, run the following:

```shell
# Installing with the `--upgrade` flag ensures you'll get the latest version.
pip install --user --upgrade tf-agents-nightly  # depends on tf-nightly
# `--force-reinstall helps guarantee the right version.
pip install --user --force-reinstall tf-nightly
pip install --user --force-reinstall tfp-nightly
```

### From GitHub

After cloning the repository, the dependencies can be installed by running
`pip install -e .[tests]`. TensorFlow needs to be installed independently:
`pip install --user tf-nightly`.

<a id='Contributing'></a>
## Contributing

We're eager to collaborate with you! See [`CONTRIBUTING.md`](CONTRIBUTING.md)
for a guide on how to contribute. This project adheres to TensorFlow's
[code of conduct](CODE_OF_CONDUCT.md). By participating, you are expected to
uphold this code.

<a id='Releases'></a>
## Releases

TF Agents does both stable and nightly releases. The nightly releases often are
fine but can have issues to to upstream libraries being in flux. The table below
lists the stable releases of TF Agents to help users that may be locked into a
specific version of TensorFlow or other related supporting. TensorFlow version
are the versions of TensorFlow tested with the build, other version might work
but were not tested.

| Release  | Branch / Tag      | TensorFlow Version |
| -------- | ----------- | ------------------ |
| Nightly  | [master](https://github.com/tensorflow/agents) | tf-nightly         |
| 0.3.0    | [v0.3.0](https://github.com/tensorflow/agents/tree/v0.3.0) | 1.15.0 and 2.0.0   |

Examples of installing nightly, most recent stable, and a specific version of
TF-Agents:

```bash
# Stable
pip install tf-agents

# Nightly
pip install tf-agents-nightly

# Specific version
pip install tf-agents==0.3.0

```


<a id='Principles'></a>
## Principles

This project adheres to [Google's AI principles](PRINCIPLES.md).
By participating, using or contributing to this project you are expected to
adhere to these principles.

<a id='Citation'></a>
## Citation

If you use this code please cite it as:

```
@misc{TFAgents,
  title = {{TF-Agents}: A library for Reinforcement Learning in TensorFlow},
  author = "{Sergio Guadarrama, Anoop Korattikara, Oscar Ramirez,
    Pablo Castro, Ethan Holly, Sam Fishman, Ke Wang, Ekaterina Gonina, Neal Wu,
    Efi Kokiopoulou, Luciano Sbaiz, Jamie Smith, Gábor Bartók, Jesse Berent,
    Chris Harris, Vincent Vanhoucke, Eugene Brevdo}",
  howpublished = {\url{https://github.com/tensorflow/agents}},
  url = "https://github.com/tensorflow/agents",
  year = 2018,
  note = "[Online; accessed 25-June-2019]"
}
```

<a id='Disclaimer'></a>
## Disclaimer

This is not an official Google product.
