# TensorFlow Agents

**Reinforcement Learning with TensorFlow**

Agents makes designing, implementing and testing new RL algorithms easier, by
providing well tested modular components that can be modified and extended. It
enables fast code iteration, with good test integration and benchmarking.

To get started, we recommend checking out one of our [tutorials](/tutorials).

## Installation

TF-Agents publishes nightly and stable builds. For a list of releases read the
<a href='#Releases'>Releases</a> section. The commands below cover installing
TF-Agents stable and nightly from [pypi.org](https://pypi.org) as well as from a
GitHub clone.

### Stable

Run the commands below to install the most recent stable release (0.5.0), which
was tested with TensorFlow 2.2.x and Python3.

```bash
pip install --user tf-agents
pip install --user tensorflow==2.2.0

# To get the matching examples and colabs
git clone https://github.com/tensorflow/agents.git
cd agents
git checkout v0.5.0

```

If you want to use TF-Agents with TensorFlow 1.15 or 2.0, install version 0.3.0:

```bash
pip install tf-agents==0.3.0
# Newer versions of tensorflow-probability require newer versions of TensorFlow.
pip install tensorflow-probability==0.8.0
```

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

After cloning the repository, the dependencies can be installed by running `pip
install -e .[tests]`. TensorFlow needs to be installed independently: `pip
install --user tf-nightly`.

<a id='Contributing'></a>

## Contributing

We're eager to collaborate with you! See
[`CONTRIBUTING.md`](https://github.com/tensorflow/agents/blob/master/CONTRIBUTING.md)
for a guide on how to contribute. This project adheres to TensorFlow's
[code of conduct](https://github.com/tensorflow/agents/blob/master/CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code.

<a id='Releases'></a>

## Releases

TF Agents has stable and nightly releases. The nightly releases are often fine
but can have issues due to upstream libraries being in flux. The table below
lists the version(s) of TensorFlow tested with each TF Agents' release to help
users that may be locked into a specific version of TensorFlow.

Release | Branch / Tag                                               | TensorFlow Version
------- | ---------------------------------------------------------- | ------------------
Nightly | [master](https://github.com/tensorflow/agents)             | tf-nightly
0.5.0   | [v0.5.0](https://github.com/tensorflow/agents/tree/v0.5.0) | 2.2.0
0.4.0   | [v0.4.0](https://github.com/tensorflow/agents/tree/v0.4.0) | 2.1.0
0.3.0   | [v0.3.0](https://github.com/tensorflow/agents/tree/v0.3.0) | 1.15.0 and 2.0.0

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

This project adheres to
[Google's AI principles](https://github.com/tensorflow/agents/blob/master/PRINCIPLES.md).
By participating, using or contributing to this project you are expected to
adhere to these principles.

<a id='Citation'></a>

## Citation

If you use this code, please cite it as:

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
