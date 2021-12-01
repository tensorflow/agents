# PPOClipAgent and PPOKLPenaltyAgent

`PPOClipAgent` and `PPOKLPenaltyAgent` implement the PPO algorithms from
["Proximal Policy Optimization Algorithms"](https://arxiv.org/abs/1707.06347)
(Schulman, 17). PPO is an on-policy policy gradient algorithm that removes the
incentives for the new policy to get far from the old policy. It is shown to
achieve better sample complexity than previous methods.

We support both the clipping version and the KL penalty version of PPO mentioned
in (Schulman, 17). The clipping version of the algorithm (`PPOClipAgent`) tends
to perform better in most environments.

Our PPO agents support custom environments with discrete and continuous action
spaces. Both the actor and the value networks support RNNs.


# Examples

Here is a simple example to train and evaluate PPOClipAgent in the HalfCheetah
environment:

```shell
$  python tf_agents/examples/ppo/schulman17/ppo_clip_train_eval.py \
     --root_dir=$HOME/tmp/ppo/HalfCheetah-v2/ \
     --gin_file=tf_agents/examples/ppo/schulman17/configs/half_cheetah.gin \
     --alsologtostderr

# To view progress in TensorBoard use another terminal and execute.
$  tensorboard --logdir $HOME/tmp/ppo/HalfCheetah-v2/
```

We set default hyperparameters according to the PPO paper. We provide gin files
to config train_eval hyperparameter arguments for several gym-mujoco
environments in the [configs](https://github.com/tensorflow/agents/tree/master/tf_agents/examples/ppo/schulman17/configs)
directory.

# Validation and Performance

To validate TF-Agents’ PPO implementation, we ran a series of experiments for
the environments referenced in
["Proximal Policy Optimization Algorithms"](https://arxiv.org/abs/1707.06347).
The table below is a summary of the results listing the mean and median of 3
experiments after 1 million environment steps with links to
[tensorboard.dev](https://tensorboard.dev/) for maximum utility.

Env                     | Mean@1M  | Median@1M | Tensorboard
----------------------- | -------- | ------- |-----------
HalfCheetah-v2 (Mujoco) | 4,239    | 4,989 | [tensorboard.dev](https://tensorboard.dev/experiment/gKfVNDIMReWQbBmESMEsyg/#scalars&runSelectionState=eyJoYWxmX2NoZWV0YWhfMDAvdHJhaW4iOmZhbHNlLCJoYWxmX2NoZWV0YWhfMDEvdHJhaW4iOmZhbHNlLCJoYWxmX2NoZWV0YWhfMDIvdHJhaW4iOmZhbHNlfQ%3D%3D&_smoothingWeight=0)
Hopper-v2 (Mujoco)      | 2,530   | 3,181 | [tensorboard.dev](https://tensorboard.dev/experiment/LxSGDZwaSjOtkX4jgQrAuQ/#scalars&runSelectionState=eyJob3BwZXJfMDAvdHJhaW4iOmZhbHNlLCJob3BwZXJfMDEvdHJhaW4iOmZhbHNlLCJob3BwZXJfMDIvdHJhaW4iOmZhbHNlfQ%3D%3D&_smoothingWeight=0)
Walker2d-v2 (Mujoco)    | 2,971   | 3,076 | [tensorboard.dev](https://tensorboard.dev/experiment/Qfqo0U1AQLGtpWPAWHB7sA/#scalars&runSelectionState=eyJ3YWxrZXJfMmRfMDAvdHJhaW4iOmZhbHNlLCJ3YWxrZXJfMmRfMDEvdHJhaW4iOmZhbHNlLCJ3YWxrZXJfMmRfMDIvdHJhaW4iOmZhbHNlfQ%3D%3D&_smoothingWeight=0)
Reacher-v2 (Mujoco)     | -7.71  | -7.81 | [tensorboard.dev](https://tensorboard.dev/experiment/t6v8GGclQeyugJe6H9a90Q/#scalars&runSelectionState=eyJyZWFjaGVyXzAwL3RyYWluIjpmYWxzZSwicmVhY2hlcl8wMS90cmFpbiI6ZmFsc2UsInJlYWNoZXJfMDIvdHJhaW4iOmZhbHNlfQ%3D%3D&_smoothingWeight=0)
Swimmer-v2 (Mujoco)     | 46.87  | 47.02 | [tensorboard.dev](https://tensorboard.dev/experiment/EY1XdwceRWG5r6kbGJqR4w/#scalars&runSelectionState=eyJzd2ltbWVyXzAwL3RyYWluIjpmYWxzZSwic3dpbW1lcl8wMS90cmFpbiI6ZmFsc2UsInN3aW1tZXJfMDIvdHJhaW4iOmZhbHNlfQ%3D%3D&_smoothingWeight=0)
InvertedPendulum-v2 (Mujoco)     |  1,000  | 1,000 | [tensorboard.dev](https://tensorboard.dev/experiment/dPdSVdfaQYaSki1wCGXIPg/#scalars&runSelectionState=eyJpbnZlcnRlZF9wZW5kdWx1bV8wMi90cmFpbiI6ZmFsc2UsImludmVydGVkX3BlbmR1bHVtXzAxL3RyYWluIjpmYWxzZSwiaW52ZXJ0ZWRfcGVuZHVsdW1fMDAvdHJhaW4iOmZhbHNlfQ%3D%3D&_smoothingWeight=0)
InvertedDoublePendulum-v2 (Mujoco)| 9,353  | 9,356 | [tensorboard.dev](https://tensorboard.dev/experiment/37xuwv5QTwmf7Dqp1PYSvw/#scalars&runSelectionState=eyJpbnZlcnRlZF9kb3VibGVfcGVuZHVsdW1fMDIvdHJhaW4iOmZhbHNlLCJpbnZlcnRlZF9kb3VibGVfcGVuZHVsdW1fMDEvdHJhaW4iOmZhbHNlLCJpbnZlcnRlZF9kb3VibGVfcGVuZHVsdW1fMDAvdHJhaW4iOmZhbHNlfQ%3D%3D&_smoothingWeight=0)

The sections that follow contain figures showing the mean along with min/max
ranges for each of 3 runs of the environments tested along with details of the
test environment and commands used. Due to all of the necessary libraries, we
created a
[Docker](https://github.com/tensorflow/agents/tree/master/tools/docker) for
TF-Agents that includes [MuJoCo](http://www.mujoco.org/). Using the Docker and
executing an experiment is detailed in the
[Reproduce results](#reproduce-results) section. The commands should also work
on a bare metal workstation setup as well.

In order to execute ~1M environment steps, we run 489 iterations
(`--num_iterations=489`) which results in 1,001,472 environment steps. Each
iteration results in 320 training steps (or 320 gradient updates, this is
calulated from environemnt_steps * num_epochs / minibatch_size) and 2,048
environment steps. Thus 489 *2,048 = 1,001,472 environment steps and 489 * 320 =
156,480 training steps.

The graphs below were generated with [Seaborn](https://seaborn.pydata.org/) and
our own [script](https://github.com/tensorflow/agents/blob/master/tools/graph_builder.py)
that reads the same logs processed by Tensorboard.


## HalfCheetah-v2 (MuJoCo)

![alt_text](https://raw.githubusercontent.com/tensorflow/agents/master/docs/images/ppo_readme/halfcheetah-v2_graph.png "HalfCheetah-v2 Mean and min/max graph.")

```shell
$  python3 tf_agents/examples/ppo/schulman17/ppo_clip_train_eval.py \
             --root_dir=./logs/ppo/half_cheetah_00/ \
             --gin_file=tf_agents/examples/ppo/schulman17/configs/half_cheetah.gin \
             --eval_interval=10000 \
             --num_iterations=489 \
             --alsologtostderr

```


## Hopper-v2 (MuJoCo)

![alt_text](https://raw.githubusercontent.com/tensorflow/agents/master/docs/images/ppo_readme/hopper-v2_graph.png "HalfCheetah-v2 Mean and min/max graph.")

```shell
$  python3 tf_agents/examples/ppo/schulman17/ppo_clip_train_eval.py \
             --root_dir=./logs/ppo/hopper_00/ \
             --gin_file=tf_agents/examples/ppo/schulman17/configs/hopper.gin \
             --eval_interval=10000 \
             --num_iterations=489 \
             --alsologtostderr

```

## Walker2d-v2 (MuJoCo)

![alt_text](https://raw.githubusercontent.com/tensorflow/agents/master/docs/images/ppo_readme/walker2d-v2_graph.png "HalfCheetah-v2 Mean and min/max graph.")

```shell
$  python3 tf_agents/examples/ppo/schulman17/ppo_clip_train_eval.py \
             --root_dir=./logs/ppo/walker_2d_00/ \
             --gin_file=tf_agents/examples/ppo/schulman17/configs/walker_2d.gin \
             --eval_interval=10000 \
             --num_iterations=489 \
             --alsologtostderr

```

## Reacher-v2 (MuJoCo)

![alt_text](https://raw.githubusercontent.com/tensorflow/agents/master/docs/images/ppo_readme/reacher-v2_graph.png "HalfCheetah-v2 Mean and min/max graph.")

```shell
$  python3 tf_agents/examples/ppo/schulman17/ppo_clip_train_eval.py \
             --root_dir=./logs/ppo/reacher_00/ \
             --gin_file=tf_agents/examples/ppo/schulman17/configs/reacher.gin \
             --eval_interval=10000 \
             --num_iterations=489 \
             --alsologtostderr

```

## Swimmer-v2 (MuJoCo)

![alt_text](https://raw.githubusercontent.com/tensorflow/agents/master/docs/images/ppo_readme/swimmer-v2_graph.png "HalfCheetah-v2 Mean and min/max graph.")

```shell
$  python3 tf_agents/examples/ppo/schulman17/ppo_clip_train_eval.py \
             --root_dir=./logs/ppo/swimmer_00/ \
             --gin_file=tf_agents/examples/ppo/schulman17/configs/swimmer.gin \
             --eval_interval=10000 \
             --num_iterations=489 \
             --alsologtostderr

```


## InvertedPendulum-v2 (MuJoCo)

![alt_text](https://raw.githubusercontent.com/tensorflow/agents/master/docs/images/ppo_readme/invertedpendulum-v2_graph.png "HalfCheetah-v2 Mean and min/max graph.")

```shell
$  python3 tf_agents/examples/ppo/schulman17/ppo_clip_train_eval.py \
             --root_dir=./logs/ppo/inverted_pendulum_00/ \
             --gin_file=tf_agents/examples/ppo/schulman17/configs/inverted_pendulum.gin \
             --eval_interval=10000 \
             --num_iterations=489 \
             --alsologtostderr

```

## InvertedDoublePendulum-v2 (MuJoCo)

![alt_text](https://raw.githubusercontent.com/tensorflow/agents/master/docs/images/ppo_readme/inverteddoublependulum-v2_graph.png "HalfCheetah-v2 Mean and min/max graph.")

```shell
$  python3 tf_agents/examples/ppo/schulman17/ppo_clip_train_eval.py \
             --root_dir=./logs/ppo/inverted_double_pendulum_00/ \
             --gin_file=tf_agents/examples/ppo/schulman17/configs/inverted_double_pendulum.gin \
             --eval_interval=10000 \
             --num_iterations=489 \
             --alsologtostderr

```


## Reproduce results

To reproduce the numbers we suggest using the commands below, which is using the
`head` of TF-Agents. Unless specified in the details of section, each experiment
was run on a Google Cloud `n1-standard-16` instance.

**Step 1:** Create a
[Google Deep Learning VM instance](https://cloud.google.com/ai-platform/deep-learning-vm/docs/tensorflow_start_instance).
The only things used from the Deep Learning VM is Docker and optionally
tensorboard which is used outside the Docker container. We are using the image
because it also includes NVIDIA Drivers and we use the same VM for GPU testing.

```shell
$  export IMAGE="c6-deeplearning-tf2-ent-2-3-cu110-v20200826-debian-9"
$  export ZONE=<Your Zone, e.g. europe-west2-b>
$  export PROJECT=<Your Project>
$  export INSTANCE_NAME=ppo-training

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

# It may take a couple minutes for the VM to be available.
# Login to the cloud instance.
$  gcloud compute ssh --zone $ZONE --project $PROJECT $INSTANCE_NAME
```

**Step 2:** Build a
[Docker](https://github.com/tensorflow/agents/tree/master/tools/docker) with
MuJoCo to be used for the experiment. These steps take place on the instance
that was created in step 1.

```shell
$  git clone https://github.com/tensorflow/agents.git && cd agents

# Core tf-agents docker.
$  docker build -t tf_agents/core \
     --build-arg tf_agents_pip_spec=tf-agents-nightly[reverb] \
     -f tools/docker/ubuntu_1804_tf_agents .

# Extends tf_agents/core to create a docker with MuJoCo.
$  docker build -t tf_agents/mujoco -f tools/docker/ubuntu_1804_mujoco .
```

**Step 3:** Create a tmux session and start the train and eval.

```shell
# Using tmux keeps the experiment running if the connection drops.
$  tmux new -s bench

# Runs the HalfCheetah-v2 example in the docker instance. The details sections
# contain commands for the other environments.
$  docker run --rm -it -v $(pwd):/workspace -w /workspace/ tf_agents/mujoco \
     python3 tf_agents/examples/ppo/schulman17/ppo_clip_train_eval.py \
             --root_dir=./logs/ppo/half_cheetah_00/ \
             --gin_file=tf_agents/examples/ppo/schulman17/configs/half_cheetah.gin \
             --eval_interval=10000 \
             --num_iterations=489 \
             --alsologtostderr
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

**Addendum**

The details below are about the process used to generate the results. The
information is stored here as an audit trail and to avoid forgetting some of the
finer details. Even following this the environment will not be exact due to
versions of other libraries moving forward; but it should be close and provides
transparency.

```shell
# After checking out the code above and before building the docker execute the following:
$  git reset --hard 61949b4c50f3610e610f1b3f3597f2360fb6878c
```

`tf-nightly==2.5.0.dev20201029` was used to generate the results and this was
the exact docker build commands:

```shell

$  docker build -t tf_agents/core \
     --build-arg tf_agents_pip_spec=tf-agents-nightly[reverb]==0.7.0.dev20201107 \
     --build-arg tensorflow_pip_spec="tf-nightly==2.5.0.dev20201029" \
     -f tools/docker/ubuntu_1804_tf_agents .

# Extends tf_agents/core to create a docker with MuJoCo.
$  docker build -t tf_agents/mujoco -f tools/docker/ubuntu_1804_mujoco .

```
