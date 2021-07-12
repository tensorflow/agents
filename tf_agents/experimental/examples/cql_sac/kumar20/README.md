# CqlSacAgent

`CqlSacAgent` implements the CQL algorithm for continuous control domains from
["Conservative Q-Learning for Offline Reinforcement Learning"](https://arxiv.org/abs/2006.04779)
(Kumar, 20).


# Background

Offline RL learns skills solely from previously-collected static datasets
without any active environment interaction, which is immensely useful
when active exploration is expensive or potentially dangerous. In practice,
this is challenging since standard off-policy RL methods can fail due to
the overestimation of values induced by the distributional shift between the
dataset and learned policy.

CQL tackles this problem by learning a conservative Q-function such that the
estimated value of a policy under this learned value function lower-bounds its
true value. To achieve this, CQL augments the standard Bellman error objective
with a Q-value regularizer that minimizes Q-values on unseen actions with
overestimated values while simultaneously maximizing the expected Q-value on the
dataset.


# Datasets

CQL is evaluated on D4RL datasets with complex data distributions and difficult
control problems.

One of these benchmarks is the Ant Maze domain. The goal in this task is to
navigate the 8-DoF ant from a start state to a goal state. The offline dataset
consists of random motions of the ant, but no single trajectory that solves
the task. A successful algorithm needs to "stitch" together different
sub-trajectories. While prior methods (e.g. BC, SAC) perform reasonably
in the easy U-maze, they are unable to stitch trajectories in the more difficult
mazes.

CQL is the only algorithm to make non-trivial progress and obtain >50%
and >14% success rates on medium and large mazes. This is because constraining
the learned policy to the dataset, as done in prior methods, tends to be overly
conservative. CQL has a "value-aware" regularizer which learns low Q-values for
unseen actions but avoids over-conservatism, so it does not need to constrain
actions to the data.

The benchmarks are described in detail in the [D4RL paper](https://arxiv.org/abs/2004.07219).

# Generate TF-Agents datasets

Here is an example to generate a dataset for the Antmaze-Medium-Play D4RL environment.

The script writes a TFRecord dataset to $DATA_ROOT_DIR/$ENV_NAME across $NUM_REPLICAS shards.

```shell
$  NUM_REPLICAS=1
$  DATA_ROOT_DIR=$HOME/tmp/d4rl_dataset
$  ENV_NAME=antmaze-medium-play-v0
$  python tf_agents/experimental/examples/cql_sac/kumar20/dataset/dataset_generator.py \
    --replicas=$NUM_REPLICAS --env_name=$ENV_NAME --root_dir=$DATA_ROOT_DIR
```

# Examples

Here is a simple example to train and evaluate CqlSacAgent on the
Antmaze-Medium-Play environment, using the dataset generated above:

```shell
$  TRAIN_EVAL_ROOT_DIR=$HOME/tmp/cql_sac/$ENV_NAME
$  python tf_agents/experimental/examples/cql_sac/kumar20/cql_sac_train_eval.py \
     --env_name=$ENV_NAME \
     --root_dir=$TRAIN_EVAL_ROOT_DIR \
     --dataset_path=$DATA_ROOT_DIR \
     --gin_file=tf_agents/experimental/examples/cql_sac/kumar20/configs/antmaze.gin \
     --alsologtostderr

# To view progress in TensorBoard use another terminal and execute.
$  tensorboard --logdir $TRAIN_EVAL_ROOT_DIR
```

We set default hyperparameters according to the CQL paper. We provide gin files
with train_eval hyperparameter arguments in the [configs](https://github.com/tensorflow/agents/tree/master/tf_agents/experimental/examples/cql_sac/kumar20/configs)
directory.
