# CqlSacAgent

[CqlSacAgent](https://github.com/tensorflow/agents/tree/master/tf_agents/agents/cql/cql_sac_agent.py)  implements the CQL algorithm for continuous control domains from
["Conservative Q-Learning for Offline Reinforcement Learning"](https://arxiv.org/abs/2006.04779)
(Kumar, 20).

WARNING: Please note that installation of D4RL environments is currently having
some issues. [Installing D4RL](https://sites.google.com/view/d4rl-anonymous/) is
a prerequisite to run the CQL for D4RL.

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

# Select RLDS D4RL dataset corresponding to environment

Select the correct RLDS dataset name from [here](https://www.tensorflow.org/datasets/catalog/overview#d4rl)
corresponding to the D4RL environment you wish to use.

For example, dataset [d4rl_antmaze/medium-play-v0](https://www.tensorflow.org/datasets/catalog/d4rl_antmaze#d4rl_antmazemedium-play-v0)
dataset for the Antmaze-Medium-Play D4RL environment.

# Examples

Here is a simple example to train and evaluate CqlSacAgent on the
Antmaze-Medium-Play environment, using the dataset generated above:

```shell
$  DATASET_NAME=d4rl_antmaze/medium-play-v0
$  ENV_NAME=antmaze-medium-play-v0
$  TRAIN_EVAL_ROOT_DIR=$HOME/tmp/cql_sac/$ENV_NAME
$  python tf_agents/examples/cql_sac/kumar20/cql_sac_train_eval.py \
     --env_name=$ENV_NAME \
     --root_dir=$TRAIN_EVAL_ROOT_DIR \
     --dataset_name=$DATASET_NAME \
     --gin_file=tf_agents/examples/cql_sac/kumar20/configs/antmaze.gin \
     --alsologtostderr

# To view progress in TensorBoard use another terminal and execute.
$  tensorboard --logdir $TRAIN_EVAL_ROOT_DIR
```

# Hyperparameters
We provide gin files with train_eval hyperparameter arguments in the
[configs](https://github.com/tensorflow/agents/tree/master/tf_agents/examples/cql_sac/kumar20/configs) directory.

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


# Validation and performance
To validate TF-Agents' CQL-SAC implementation, we ran a series of experiments for
the environments referenced in the paper. The table below is a summary of the
results listing the mean of 4 experiments at the declared number of steps with
links to TensorBoard.dev for maximum utility.

|Environment              |Measurement 1|Measurement 2|Tensorboard                                                        |
|-------------------------|-------------|-------------|-------------------------------------------------------------------|
|HalfCheetah-Medium-Expert| 3,709@500K  |   8,998@1M  |[tb](https://tensorboard.dev/experiment/CwOJRZmTQYaauo1eMtZCBQ/) |
|HalfCheetah-Medium       |  4,877@500K |   4,623@1M  |[tb](https://tensorboard.dev/experiment/rdIa9L8QRh26KvVnfkFK1w/)|
|Walker2d-Medium-Expert   |  3,287@500K |   3,228@1M  |[tb](https://tensorboard.dev/experiment/tBbZEe3WT7yjfagDpn40LQ/)|
|Walker2d-Medium          |  3,665@500K |   3,686@1M  |[tb](https://tensorboard.dev/experiment/2QxrKaydSqGxrw9RcL22HA/)|
|Hopper-Medium-Expert     |  3,645@500K |   3,622@1M  |[tb](https://tensorboard.dev/experiment/WwbwLPn0Tmi9HygN49DcEQ/) |
|Hopper-Medium            |  1,669@500K |    1,614@1M |[tb](https://tensorboard.dev/experiment/L1PIoY2NRwujx78FZa3jOQ/)|
|AntMaze-Medium-Diverse   |  0.475@500K |    0.625@1M |[tb](https://tensorboard.dev/experiment/6yZsNBOHQxWcBwz3IEz51w/)  |
|AntMaze-Medium-Play      |  0.325@500K  |  0.450@1M  |[tb](https://tensorboard.dev/experiment/wg75CDJOTDmaJ54Iq0LVvw/)|
|AntMaze-Large-Diverse    |  0.0750@500K |  0.1750@1M |[tb](https://tensorboard.dev/experiment/P77CBaELQjOO1emWmIqtMQ/)|
|AntMaze-Large-Play       |  0.0750@500K |  0.1500@1M |[tb](https://tensorboard.dev/experiment/zjWRY0YgT9yKAEC7rt94KA/)|

The sections that follow contain figures showing the mean along with min/max
ranges for each of 4 runs of the environments tested along with details of the
test environment and commands used. Due to all of the necessary libraries, we
created a Docker for TF-Agents that includes [MuJoCo](http://www.mujoco.org/) and
[D4RL](https://github.com/rail-berkeley/d4rl). Using the Docker and executing
an experiment is detailed in the [reproduce results](#reproduce-results) section.
The commands should also work on a bare metal workstation setup as well.

Before training each of the below, run dataset generation for that environment:

## HalfCheetah-Medium-Expert (MuJoCo)

![alt_text](https://raw.githubusercontent.com/tensorflow/agents/master/docs/images/cql_sac_readme/halfcheetah-medium-expert-v0_graph.png "halfcheetah-medium-expert-v0 Mean and min/max graph.")

```shell
$  ENV_NAME=halfcheetah-medium-expert-v0
$  TRAIN_EVAL_ROOT_DIR=./tmp/cql_sac/$ENV_NAME
$  DATASET_NAME=d4rl_mujoco_halfcheetah/v0-medium-expert
$  python tf_agents/examples/cql_sac/kumar20/cql_sac_train_eval.py \
     --env_name=$ENV_NAME \
     --root_dir=$TRAIN_EVAL_ROOT_DIR \
     --dataset_name=$DATASET_NAME \
     --gin_file=tf_agents/examples/cql_sac/kumar20/configs/mujoco_medium_expert.gin \
     --alsologtostderr
```

## HalfCheetah-Medium (MuJoCo)

![alt_text](https://raw.githubusercontent.com/tensorflow/agents/master/docs/images/cql_sac_readme/halfcheetah-medium-v0_graph.png "halfcheetah-medium-v0 Mean and min/max graph.")

```shell
$  ENV_NAME=halfcheetah-medium-v0
$  TRAIN_EVAL_ROOT_DIR=./tmp/cql_sac/$ENV_NAME
$  python tf_agents/examples/cql_sac/kumar20/cql_sac_train_eval.py \
     --env_name=$ENV_NAME \
     --root_dir=$TRAIN_EVAL_ROOT_DIR \
     --dataset_name=$DATASET_NAME \
     --gin_file=tf_agents/examples/cql_sac/kumar20/configs/mujoco_medium.gin \
     --alsologtostderr
```

## Hopper-Medium-Expert (MuJoCo)

![alt_text](https://raw.githubusercontent.com/tensorflow/agents/master/docs/images/cql_sac_readme/hopper-medium-expert-v0_graph.png "hopper-medium-expert-v0 Mean and min/max graph.")

```shell
$  ENV_NAME=hopper-medium-expert-v0
$  TRAIN_EVAL_ROOT_DIR=./tmp/cql_sac/$ENV_NAME
$  python tf_agents/examples/cql_sac/kumar20/cql_sac_train_eval.py \
     --env_name=$ENV_NAME \
     --root_dir=$TRAIN_EVAL_ROOT_DIR \
     --dataset_name=$DATASET_NAME \
     --gin_file=tf_agents/examples/cql_sac/kumar20/configs/mujoco_medium_expert.gin \
     --alsologtostderr
```

## Hopper-Medium (MuJoCo)

![alt_text](https://raw.githubusercontent.com/tensorflow/agents/master/docs/images/cql_sac_readme/hopper-medium-v0_graph.png "hopper-medium-v0 Mean and min/max graph.")

```shell
$  ENV_NAME=hopper-medium-v0
$  TRAIN_EVAL_ROOT_DIR=./tmp/cql_sac/$ENV_NAME
$  python tf_agents/examples/cql_sac/kumar20/cql_sac_train_eval.py \
     --env_name=$ENV_NAME \
     --root_dir=$TRAIN_EVAL_ROOT_DIR \
     --dataset_name=$DATASET_NAME \
     --gin_file=tf_agents/examples/cql_sac/kumar20/configs/mujoco_medium.gin \
     --alsologtostderr
```

## Walker2d-Medium-Expert (MuJoCo)

![alt_text](https://raw.githubusercontent.com/tensorflow/agents/master/docs/images/cql_sac_readme/walker2d-medium-expert-v0_graph.png "walker2d-medium-expert-v0 Mean and min/max graph.")

```shell
$  ENV_NAME=walker2d-medium-expert-v0
$  TRAIN_EVAL_ROOT_DIR=./tmp/cql_sac/$ENV_NAME
$  python tf_agents/examples/cql_sac/kumar20/cql_sac_train_eval.py \
     --env_name=$ENV_NAME \
     --root_dir=$TRAIN_EVAL_ROOT_DIR \
     --dataset_name=$DATASET_NAME \
     --gin_file=tf_agents/examples/cql_sac/kumar20/configs/mujoco_medium_expert.gin \
     --alsologtostderr
```

## Walker2d-Medium (MuJoCo)

![alt_text](https://raw.githubusercontent.com/tensorflow/agents/master/docs/images/cql_sac_readme/walker2d-medium-v0_graph.png "walker2d-medium-v0 Mean and min/max graph.")

```shell
$  ENV_NAME=walker2d-medium-v0
$  TRAIN_EVAL_ROOT_DIR=./tmp/cql_sac/$ENV_NAME
$  python tf_agents/examples/cql_sac/kumar20/cql_sac_train_eval.py \
     --env_name=$ENV_NAME \
     --root_dir=$TRAIN_EVAL_ROOT_DIR \
     --dataset_name=$DATASET_NAME \
     --gin_file=tf_agents/examples/cql_sac/kumar20/configs/mujoco_medium.gin \
     --alsologtostderr
```

## AntMaze-Medium-Diverse

![alt_text](https://raw.githubusercontent.com/tensorflow/agents/master/docs/images/cql_sac_readme/antmaze-medium-diverse-v0_graph.png "antmaze-medium-diverse-v0 Mean and min/max graph.")

```shell
$  ENV_NAME=antmaze-medium-diverse-v0
$  TRAIN_EVAL_ROOT_DIR=./tmp/cql_sac/$ENV_NAME
$  python tf_agents/examples/cql_sac/kumar20/cql_sac_train_eval.py \
     --env_name=$ENV_NAME \
     --root_dir=$TRAIN_EVAL_ROOT_DIR \
     --dataset_name=$DATASET_NAME \
     --gin_file=tf_agents/examples/cql_sac/kumar20/configs/antmaze.gin \
     --alsologtostderr
```


## AntMaze-Medium-Play

![alt_text](https://raw.githubusercontent.com/tensorflow/agents/master/docs/images/cql_sac_readme/antmaze-medium-play-v0_graph.png "antmaze-medium-play-v0 Mean and min/max graph.")

```shell
$  ENV_NAME=antmaze-medium-play-v0
$  TRAIN_EVAL_ROOT_DIR=./tmp/cql_sac/$ENV_NAME
$  python tf_agents/examples/cql_sac/kumar20/cql_sac_train_eval.py \
     --env_name=$ENV_NAME \
     --root_dir=$TRAIN_EVAL_ROOT_DIR \
     --dataset_name=$DATASET_NAME \
     --gin_file=tf_agents/examples/cql_sac/kumar20/configs/antmaze.gin \
     --alsologtostderr
```

## AntMaze-Large-Diverse

![alt_text](https://raw.githubusercontent.com/tensorflow/agents/master/docs/images/cql_sac_readme/antmaze-large-diverse-v0_graph.png "antmaze-large-diverse-v0 Mean and min/max graph.")

```shell
$  ENV_NAME=antmaze-large-diverse-v0
$  TRAIN_EVAL_ROOT_DIR=./tmp/cql_sac/$ENV_NAME
$  python tf_agents/examples/cql_sac/kumar20/cql_sac_train_eval.py \
     --env_name=$ENV_NAME \
     --root_dir=$TRAIN_EVAL_ROOT_DIR \
     --dataset_name=$DATASET_NAME \
     --gin_file=tf_agents/examples/cql_sac/kumar20/configs/antmaze.gin \
     --alsologtostderr
```


## AntMaze-Large-Play

![alt_text](https://raw.githubusercontent.com/tensorflow/agents/master/docs/images/cql_sac_readme/antmaze-large-play-v0_graph.png "antmaze-large-play-v0 Mean and min/max graph.")

```shell
$  ENV_NAME=antmaze-large-play-v0
$  TRAIN_EVAL_ROOT_DIR=./tmp/cql_sac/$ENV_NAME
$  python tf_agents/examples/cql_sac/kumar20/cql_sac_train_eval.py \
     --env_name=$ENV_NAME \
     --root_dir=$TRAIN_EVAL_ROOT_DIR \
     --dataset_name=$DATASET_NAME \
     --gin_file=tf_agents/examples/cql_sac/kumar20/configs/antmaze.gin \
     --alsologtostderr
```

# Reproduce results

We ran a series of experiments for the environments referenced in the CQL paper.
To reproduce our results we suggest using the commands below, which is using the
`head` of TF-Agents.

**Step 1:** Create a
[Google Deep Learning VM instance](https://cloud.google.com/ai-platform/deep-learning-vm/docs/tensorflow_start_instance).
The only things used from the Deep Learning VM is Docker and optionally
tensorboard which is used outside the Docker container. We are using the image
because it also includes NVIDIA Drivers and we use the same VM for GPU testing.

```shell
$  export IMAGE="c6-deeplearning-tf2-ent-2-3-cu110-v20200826-debian-9"
$  export ZONE=<Your Zone, e.g. europe-west2-b>
$  export PROJECT=<Your Project>
$  export INSTANCE_NAME=cql-sac-training

# 16 vCPU
$  gcloud compute instances create $INSTANCE_NAME \
    --project=$PROJECT \
    --zone=$ZONE \
    --machine-type=n1-standard-16 \
    --maintenance-policy=TERMINATE \
    --image=$IMAGE \
    --image-project=ml-images \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --scopes=https://www.googleapis.com/auth/cloud-platform

# Login to the cloud instance.
$  gcloud compute ssh --zone $ZONE --project $PROJECT $INSTANCE_NAME
```

**Step 2:** Build a
[Docker](https://github.com/tensorflow/agents/tree/master/tools/docker) with
MuJoCo and D4RL to be used for the experiment. These steps take place on the
cloud instance created in step 1.

```shell
$  git clone https://github.com/tensorflow/agents.git && cd agents

# Core tf-agents docker.
$  docker build -t tf_agents/core \
     --build-arg tf_agents_pip_spec=tf-agents-nightly[reverb] \
     -f tools/docker/ubuntu_tf_agents tools/docker/

# Extends tf_agents/core to create a docker with MuJoCo.
$  docker build -t tf_agents/mujoco -f tools/docker/ubuntu_mujoco_oss tools/docker

# Extends tf_agents/mujoco to create a docker with D4RL.
$  docker build -t tf_agents/mujoco/d4rl -f tools/docker/ubuntu_d4rl tools/docker
```

**Step 3:** Create a tmux session and start the train and eval.

```shell
# Using tmux keeps the experiment running if the connection drops.
$  tmux new -s bench

# Runs the Antmaze-Medium-Play example in the docker image. Includes data generation and training.
$  export NUM_REPLICAS=10
$  export DATASET_NAME=d4rl_mujoco_halfcheetah/v0-medium-expert
$  export ENV_NAME=halfcheetah-medium-v0

# Starts the training.
# Run `docker run --rm -it --gpus all -v` for GPU support.
$  docker run --rm -it -v $(pwd):/workspace -w /workspace/ \
    tf_agents/mujoco/d4rl \
     bash -c "python3 -m tf_agents.experimental.examples.cql_sac.kumar20.cql_sac_train_eval "\
"--env_name=$ENV_NAME --root_dir=./log/$ENV_NAME --dataset_name=$DATASET_NAME "\
"--gin_file=tf_agents/examples/cql_sac/kumar20/configs/mujoco_medium.gin --alsologtostderr"
```

**Step 4 (Optional):** Use [tensorboard.dev](https://tensorboard.dev/) to track
progress.

```shell
# To upload the results to tensorboard.dev (public) exit out of the tmux session
# and start another session to upload the results. This command is run outside
# of Docker and assumes tensorboard is installed on the host system.
# Hint: `ctrl + b` followed by `d` will detach from the session.
$  tmux new -s tensorboard
$  tensorboard dev upload --logdir=./logs/
```

# Citation
If you use this code, please cite it using [this guideline](https://github.com/tensorflow/agents#citation).
