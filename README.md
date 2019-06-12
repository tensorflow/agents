# TF-Agents: A library for Reinforcement Learning in TensorFlow

*NOTE:* Current TF-Agents pre-release is under active development and
interfaces may change at any time. Feel free to provide feedback and comments.

To get started, we recommend checking out one of our Colab tutorials. If you
need an intro to RL (or a quick recap),
[start here](tf_agents/colabs/0_intro_rl.ipynb). Otherwise, check out our
[DQN tutorial](tf_agents/colabs/1_dqn_tutorial.ipynb) to get an agent up and
running in the Cartpole environment.

## Table of contents

<a href="#Agents">Agents</a><br>
<a href="#Tutorials">Tutorials</a><br>
<a href='#Examples'>Examples</a><br>
<a href="#Installation">Installation</a><br>
<a href='#Contributing'>Contributing</a><br>
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

<a id='Examples'></a>
## Examples
End-to-end examples training agents can be found under each agent directory.
e.g.:

* DQN: [`tf_agents/agents/dqn/examples/v1/train_eval_gym.py`](https://github.com/tensorflow/agents/tree/master/tf_agents/agents/dqn/examples/v1/train_eval_gym.py)

<a id='Installation'></a>
## Installation

To install the latest version, use nightly builds of TF-Agents under the pip package
`tf-agents-nightly`, which requires you install on one of `tf-nightly` and
`tf-nightly-gpu` and also `tfp-nightly`.
Nightly builds include newer features, but may be less stable than the versioned releases.

To install the nightly build version, run the following:

```shell
# Installing with the `--upgrade` flag ensures you'll get the latest version.
pip install --user --upgrade tf-agents-nightly  # depends on tf-nightly
```

If you clone the repository you will still need a `tf-nightly` installation. You can then run `pip install -e .[tests]` from the agents directory to get dependencies to run tests.

<a id='Contributing'></a>
## Contributing

We're eager to collaborate with you! See [`CONTRIBUTING.md`](CONTRIBUTING.md)
for a guide on how to contribute. This project adheres to TensorFlow's
[code of conduct](CODE_OF_CONDUCT.md). By participating, you are expected to
uphold this code.

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
    Pablo Castro, Ethan Holly, Sam Fishman, Ke Wang, Ekaterina Gonina,
    Chris Harris, Vincent Vanhoucke, Eugene Brevdo}",
  howpublished = {\url{https://github.com/tensorflow/agents}},
  url = "https://github.com/tensorflow/agents",
  year = 2018,
  note = "[Online; accessed 30-November-2018]"
}
```

<a id='Disclaimer'></a>
## Disclaimer

This is not an official Google product.
