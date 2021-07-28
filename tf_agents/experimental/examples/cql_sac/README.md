# CqlSacAgent

[CqlSacAgent](https://github.com/tensorflow/agents/tree/master/tf_agents/agents/cql/cql_sac_agent.py)  implements the CQL algorithm for continuous control domains from
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

# Hyperparameters
We provide gin files with train_eval hyperparameter arguments in the
[configs](https://github.com/tensorflow/agents/tree/master/tf_agents/experimental/examples/cql_sac/kumar20/configs) directory.

<table>
  <tr>
    <td></td>
    <th scope="col">AntMaze (all difficulty levels)</th>
    <th scope="col">MuJoCo (medium)</th>
    <th scope="col">MuJoCo (medium-expert)</th>
  </tr>
  <tr>
    <th scope="row">use_lagrange_cql_alpha</th>
    <td>True</td>
    <td>False</td>
    <td>False</td>
  </tr>
  <tr>
    <th scope="row">cql_alpha</th>
    <td>5.0</td>
    <td>0.1</td>
    <td>1.0</td>
  </tr>
  <tr>
    <th scope="row">cql_tau</th>
    <td>5.0</td>
    <td>N/A</td>
    <td>N/A</td>
  </tr>
  <tr>
    <th scope="row">softmax_temperature</th>
    <td>50.0</td>
    <td>1.0</td>
    <td>1.0</td>
  </tr>
  <tr>
    <th scope="row">reward_scale_factor</th>
    <td>4.0</td>
    <td>0.1</td>
    <td>0.1</td>
  </tr>
  <tr>
    <th scope="row">reward_shift</th>
    <td>-0.5</td>
    <td>0</td>
    <td>0</td>
  </tr>
  <tr>
    <th scope="row">reward_noise_variance</th>
    <td>0.1</td>
    <td>0</td>
    <td>0</td>
  </tr>
  <tr>
    <th scope="row">actor_learning_rate</th>
    <td>1e-4</td>
    <td>3e-4</td>
    <td>3e-4</td>
  </tr>
  <tr>
    <th scope="row">critic_learning_rate</th>
    <td>3e-4</td>
    <td>3e-4</td>
    <td>3e-4</td>
  </tr>
  <tr>
    <th scope="row">action_clipping</th>
    <td>None</td>
    <td>(-0.995, 0.995)</td>
    <td>(-0.995, 0.995)</td>
  </tr>
  <tr>
    <th scope="row">log_cql_alpha_clipping</th>
    <td>(-1, 100)</td>
    <td>None</td>
    <td>None</td>
  </tr>
  <tr>
    <th scope="row">bc_steps</th>
    <td>10,000</td>
    <td>10,000</td>
    <td>10,000</td>
  </tr>
</table>

We'll highlight some hyperparameters which might be useful guidelines when using TF-Agents CQL-SAC. Hyperparameters are further explained in Appendix F of the CQL paper.

## cql_alpha
The most important hyperparameter to tune is cql_alpha. With a higher cql_alpha value, CQL can further push down the learned Q-value at out-of-distribution actions, correcting for erroneous overestimation error.

*  As more data becomes available and sampling error decreases, smaller cql_alpha values are enough to provide an improvement over the behavior policy.
*  cql_alpha also depends on the coverage of the state space. If the data distribution covers a small region of the state-action space, it is recommended to use a larger value (i.e., be more conservative) vs a smaller value of cql_alpha when the dataset covers a larger portion of the state-action space. For example, MuJoCo medium-expert datasets have a narrower scope than medium datasets, so a higher cql_alpha value is used (1.0 vs 0.1).

## cql_tau
To avoid specifying a manual value of cql_alpha, another option is to instead specify a target value of the CQL loss that should be achieved and maintained through training. We call this version **CQL-Lagrange**. If using CQL-Lagrange, cql_tau is the most important hyperparameter.

If the expected difference in Q-values is lower than the specified threshold cql_tau, cql_alpha will adjust to be close to 0. If the difference in Q-values is higher than cql_tau, then cql_alpha is likely to take on high values and thus more aggressively penalize Q-values. Suggested values for cql_tau are [2.0, 5.0, 10.0].

## softmax_temperature
In this implementation, we utilized hyperparameters for the base SAC algorithm that were slightly different from the paper (e.g., see *reward_scale_factor* below). While we would simply expect that Q-values be multiplied by the corresponding reward scale, we found that in this case, optimization can push down Q-values very aggressively. This is because the logsumexp term in CQL behaves as a mean instead of logsumexp (= soft maximum). To correct for this, we introduced a softmax_temperature parameter that weights Q-values before the logsumexp calculation and then unweights them after the logsumexp is computed.

When should you use it? If Q-values become too negative during learning (see *Debug summaries* below), even though the policy is trying to maximize the Q-function, this indicates the need for a higher softmax_temperature. This trick keeps Q-values positive and stable, which allows for CQL to work and yield higher returns.

## reward_scale_factor
Adjust this hyperparameter for environment-specific reward scaling. TF-Agents SacAgent uses a reward_scale of 0.1, so we apply the same scaling in CqlSacAgent. For AntMaze domains, we retain the scaling used by the paper.

# Debugging
## Running in behavioral cloning mode
To validate whether CQL is likely to work on your problem, especially if cql_loss has not been decreasing, initialize CqlSacAgent with `bc_debug_mode=True`. This runs the agent in a behavioral cloning mode where the critic loss only depends on CQL loss, and the learning algorithm performs an EBM-style (energy based model) behavior cloning. cql_loss should decrease and approach zero. If not, CQL will not work or you need to adjust hyperparameters and retry.

## Debug summaries
Initialize the agent with `debug_summaries=True` to track the Q-values. If Q-values become too negative, then CQL will not work well. This debugging enabled us to discover softmax_temperature when we saw that Q-values were decreasing too much in initial MuJoCo experiments.

# Citation
If you use this code, please cite it using [this guideline](/third_party/py/tf_agents/opensource/g3doc/README.oss.md#citation).
