# Bandits in TF-Agents

The <b>Multi-Armed Bandit problem (MAB)</b> is a special case of Reinforcement
Learning (RL): an agent collects rewards in an environment by taking some
actions after observing some state of the environment. The main difference
between general RL and MAB is that in MAB, we assume that the action taken by
the agent does not influence the next state of the environment. Therefore,
agents do not model state transitions, credit rewards to past actions, or
"plan ahead" to get to reward-rich states. Due to this very fact, the notion of
episodes is not used in MAB, unlike in general RL.

In many bandits use cases, the state of the environment is observed. These are
known as <b>contextual bandits</b> problems, and can be thought of as a
generalization of multi-armed bandits where the agent has access to additional
context in each round.

To get started with Bandits in TF-Agents, we recommend checking our [bandits tutorial](https://github.com/tensorflow/agents/blob/master/docs/tutorials/bandits_tutorial.ipynb).

## Agents

Currently the following algorithms are available:

* [`LinUCB`](https://github.com/tensorflow/agents/tree/master/tf_agents/bandits/agents/lin_ucb_agent.py):
[__A Contextual Bandit Approach to Personalized News Article Recommendation__ Li et al., 2010.](https://arxiv.org/abs/1003.0146)
* [`Linear Thompson Sampling`](https://github.com/tensorflow/agents/tree/master/tf_agents/bandits/agents/linear_thompson_sampling_agent.py):
[__Thompson Sampling for Contextual Bandits with Linear Payoffs__ Agrawal et al., 2013.](http://proceedings.mlr.press/v28/agrawal13.pdf)
* [`Neural Epsilon Greedy`](https://github.com/tensorflow/agents/tree/master/tf_agents/bandits/agents/neural_epsilon_greedy_agent.py):
[__Bandit Algorithms__ Lattimore et al., 2019](https://tor-lattimore.com/downloads/book/book.pdf)
* [`Neural LinUCB`](https://github.com/tensorflow/agents/tree/master/tf_agents/bandits/agents/neural_linucb_agent.py):
[__Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep Networks for Thompson Sampling__ Riquelme et al., 2018]
(https://arxiv.org/abs/1802.09127)
* [`Thompson Sampling with Dropout`](https://github.com/tensorflow/agents/tree/master/tf_agents/bandits/agents/dropout_thompson_sampling_agent.py):
[__Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep Networks for Thompson Sampling__ Riquelme et al., 2018]
(https://arxiv.org/abs/1802.09127)
* [`Multi-objective neural agent`](https://github.com/tensorflow/agents/tree/master/tf_agents/bandits/agents/greedy_multi_objective_neural_agent.py):
[__Designing multi-objective multi-armed bandits algorithms: a study__  Drugan et al., 2013.](https://ieeexplore.ieee.org/document/6707036)
* [`EXP3`](https://github.com/tensorflow/agents/tree/master/tf_agents/bandits/agents/exp3_agent.py):
[__Bandit Algorithms__ Lattimore et al., 2019](https://tor-lattimore.com/downloads/book/book.pdf)


## Environments

In bandits, the environment is responsible for (i) outputting information about
the current state (aka observation or context), and (ii) outputting a reward
when receiving an action as input.

In order to test the performance of existing
and new bandit algorithms, the library provides several environments spanning
various setups such as linear or non-linear rewards functions, stationary or
non-stationary environment dynamics.
More specifically, the following environments are available:

* [Stationary](https://github.com/tensorflow/agents/tree/master/tf_agents/bandits/environments/stationary_stochastic_py_environment.py):
This environment assumes stationary functions for generating observations and
rewards.
* [Non-stationary](https://github.com/tensorflow/agents/tree/master/tf_agents/bandits/environments/non_stationary_stochastic_environment.py):
This environment has non-stationary dynamics.
* [Piecewise stationary](https://github.com/tensorflow/agents/tree/master/tf_agents/bandits/environments/piecewise_stochastic_environment.py):
This environment is non-stationary, consisting of stationary pieces.
* [Drifting](https://github.com/tensorflow/agents/tree/master/tf_agents/bandits/environments/drifting_linear_environment.py):
In this case, the environment is also non-stationary and its dynamics are
slowly drifting.
* [Wheel](https://github.com/tensorflow/agents/tree/master/tf_agents/bandits/environments/wheel_py_environment.py):
This is a non-linear environment with a scalar parameter that directly controls
the difficulty of the problem.
* [Classification suite](https://github.com/tensorflow/agents/tree/master/tf_agents/bandits/environments/classification_environment.py):
Given any classification dataset wrapped as a `tf.data.Dataset`, this
environment converts it into a bandit problem.


### Regret metrics

The library also provides [TF-metrics](https://github.com/tensorflow/agents/tree/master/tf_agents/bandits/metrics/tf_metrics.py) for regret computation.
The notion of <b>regret</b> is an important one in the bandits literature and
it can be informally defined as the difference between the total expected
reward using the optimal policy and the total expected reward collected by the
agent.
Most of the environments listed above come with utilities for computing metrics
such as the regret, the percentage of suboptimal arm plays and so on.


## Examples

The library provides ready-to-use end-to-end examples for training and
evaluating various bandit agents in the
[`tf_agents/bandits/agents/examples/v2/`](https://github.com/tensorflow/agents/tree/master/tf_agents/bandits/agents/examples/v2)
directory. A few examples:

* [`Stationary linear`](https://github.com/tensorflow/agents/tree/master/tf_agents/bandits/agents/examples/v2/train_eval_stationary_linear.py):
tests different bandit agents against stationary linear environments.
* [`Wheel`](https://github.com/tensorflow/agents/tree/master/tf_agents/bandits/agents/examples/v2/train_eval_wheel.py):
tests different bandit agents against the wheel bandit environment.
* [`Drifting linear`](https://github.com/tensorflow/agents/tree/master/tf_agents/bandits/agents/examples/v2/train_eval_drifting_linear.py):
tests different bandit agents against drifting (i.e., non-stationary) linear
environments.

## Advanced functionality

#### Arm features
In some bandits use cases, each arm has its own features. For example, in
movie recommendation problems, the user features play the role of the context
and the movies play the role of the arms (aka actions). Each movie has its own
features, such as text description, metadata, trailer content features and
so on. We refer to such problems as <b>arm features problems</b>.

An example of bandit training with arm features can be found
[`here`](https://github.com/tensorflow/agents/tree/master/tf_agents/bandits/agents/examples/v2/train_eval_per_arm_stationary_linear.py).


#### Multi-metric bandits

In some bandits use cases, the "goodness" of the decisions that the agent makes
can be measured via multiple metrics. For example, when we recommend a
certain movie to a user, we can measure several metrics about this decision,
such as: whether the user clicked it, whether the user watched it, whether the
user liked it, shared it and so on. For such bandits use cases, the library
provides the following solutions.

* <b>Multi-objective optimization</b>
In case of several reward signals, a common technique is called
<i>scalarization</i>. The main idea is to combine all the input rewards signals
into a single one, which can be optimized by the vanilla bandits algorithms.
The library offers the following options for scalarization:
<ul>
  <li>Linear [Designing multi-objective multi-armed bandits algorithms: a study  Drugan et al., 2013.](https://ieeexplore.ieee.org/document/6707036)</li>
  <li>Chebyshev [Designing multi-objective multi-armed bandits algorithms: a study  Drugan et al., 2013.](https://ieeexplore.ieee.org/document/6707036)</li>
  <li>Hypervolume [Random Hypervolume Scalarizations for Provable Multi-Objective Black Box Optimization Golovin et al., 2020](https://arxiv.org/abs/2006.04655)</li>
</ul>
* <b>Constrained optimization</b>
In use cases where one metric clearly plays the role of reward metric and other
metrics can be understood as auxiliary constraint metrics, constrained
optimization may be a good fit. In this case, one can introduce the notion of
<i>action feasibility</i>, which may be context-dependent, and implies whether
an action is eligible to be selected or not in the current round (given the
current context). In the general case, the action feasibility is inferred by
evaluating expressions involving one or more of the auxiliary constraint
metrics.
The [Constraints API](https://github.com/tensorflow/agents/tree/master/tf_agents/bandits/policies/constraints.py)
unifies how all constraints are evaluated for computing the action feasibility.
A single constraint may be trainable (or not) depending on whether the action
feasibility computation is informed by a model predicting the value of the
corresponding constraint metric.

