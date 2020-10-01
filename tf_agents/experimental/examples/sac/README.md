# SACAgent

SACAgent implements the Soft Actor-Critic (SAC) algorithm from
["Soft Actor-Critic Algorithms and Applications" ](https://arxiv.org/abs/1812.05905)(Haarnoja,
18). SAC is an off-policy actor-critic deep RL algorithm. Its policy is trained
to maximize the combination of return and entropy to encourage exploration. SAC
was shown to achieve high stability and sample efficiency in various benchmarks.
It is also able to learn complex tasks such as 21-dimensional Humanoid.

SACAgent supports custom environments with continuous action spaces. Both the
actor and the critic networks support RNNs.

SACAgent also supports distributed training on multiple GPUs or TPUs when used
with the Actor-Learner API. We have a tutorial on distributed training coming
soon.

# Examples

Here is a simple example to train and evaluate SACAgent in the HalfCheetah
environment:

```shell
$  python tf_agents/experimental/examples/sac/haarnoja18/sac_train_eval.py \
     --root_dir=$HOME/tmp/sac/HalfCheetah-v2/ \
     --gin_file=tf_agents/experimental/examples/sac/haarnoja18/configs/half_cheetah.gin \
     --num_iterations=3000000 \
     --alsologtostderr

# To view progress in TensorBoard use another terminal and execute.
$  tensorboard --logdir $HOME/tmp/sac/HalfCheetah-v2/
```

We set default hyperparameters according to the SAC paper. We made minor
modifications in the number of initial collect steps, reward scaling (0.1
instead of 1.0) and the heuristic entropy target (-dim(A)/2 instead of -dim(A)).
Those modifications provide a slight benefit in the more difficult gym-mujoco
environments. Optimal reward scaling is likely to be different for other
environment suites. We provide gin files to config train_eval hyperparameter
arguments for several gym-mujoco environments in the
[configs](https://github.com/tensorflow/agents/tree/master/tf_agents/experimental/examples/sac/haarnoja18/configs)
directory.

Note: [Issue #475](https://github.com/tensorflow/agents/issues/475):
The default setup does not checkpoint the policy or data. If the pipeline gets
preempted, training and evaluation will start over. You may observe more than
one learning curve in this situation. If you enable policy checkpointing, the
learner will reload the most recent saved policy. However, because we are not
yet checkpointing the data from Reverb, we refill the table with the data
collected by the current policy. We lose the data collected with older policies.
This will result in distortions in the return that we will address in the future.

# Validation and Performance

To validate TF-Agents’ SAC implementation, we ran a series of experiments for
the environments referenced in Soft Actor-Critic Algorithms and Applications by
Haarnoja et al (2019). The table below is a summary of the results listing the
mean of 5 experiments at the declared number of steps with links to
TensorBoard.dev for maximum utility.

Env                     | Measurement 1 | Measurement 2 | Tensorboard
----------------------- | ------------- | ------------- | -----------
HalfCheetah-v2 (Mujoco) | 11,116@1M     | 13,250@3M     | [tensorboard.dev](https://tensorboard.dev/experiment/Bsb224EkTJ2ImSgyyPAjkA/#scalars&runSelectionState=eyJIYWxmQ2hlZXRhaC12Ml8wMC9ldmFsIjp0cnVlLCJIYWxmQ2hlZXRhaC12Ml8wMC90cmFpbiI6ZmFsc2UsIkhhbGZDaGVldGFoLXYyXzAxL2V2YWwiOnRydWUsIkhhbGZDaGVldGFoLXYyXzAxL3RyYWluIjpmYWxzZSwiSGFsZkNoZWV0YWgtdjJfMDIvZXZhbCI6dHJ1ZSwiSGFsZkNoZWV0YWgtdjJfMDIvdHJhaW4iOmZhbHNlLCJIYWxmQ2hlZXRhaC12Ml8wMy9ldmFsIjp0cnVlLCJIYWxmQ2hlZXRhaC12Ml8wMy90cmFpbiI6ZmFsc2UsIkhhbGZDaGVldGFoLXYyXzA0L2V2YWwiOnRydWUsIkhhbGZDaGVldGFoLXYyXzA0L3RyYWluIjpmYWxzZX0%3D)
Hopper-v2 (Mujoco)      | 2,540@1M      | n/a           | [tensorboard.dev](https://tensorboard.dev/experiment/ZMxemRYmQaK0vf6iP5PQwQ/#scalars&runSelectionState=eyJIb3BwZXItdjJfMDAvZXZhbCI6dHJ1ZSwiSG9wcGVyLXYyXzAwL3RyYWluIjpmYWxzZSwiSG9wcGVyLXYyXzAxL2V2YWwiOnRydWUsIkhvcHBlci12Ml8wMS90cmFpbiI6ZmFsc2UsIkhvcHBlci12Ml8wMi9ldmFsIjp0cnVlLCJIb3BwZXItdjJfMDIvdHJhaW4iOmZhbHNlLCJIb3BwZXItdjJfMDMvZXZhbCI6dHJ1ZSwiSG9wcGVyLXYyXzAzL3RyYWluIjpmYWxzZSwiSG9wcGVyLXYyXzA0L2V2YWwiOnRydWUsIkhvcHBlci12Ml8wNC90cmFpbiI6ZmFsc2V9")
Walker2d-v2 (Mujoco)    | 4,672@1M      | 5,823@3M      | [tensorboard.dev](https://tensorboard.dev/experiment/qW4nCOtlT52Rj6N4Mp6uwA/#scalars&runSelectionState=eyJXYWxrZXIyZC12Ml8wMC9ldmFsIjp0cnVlLCJXYWxrZXIyZC12Ml8wMC90cmFpbiI6ZmFsc2UsIldhbGtlcjJkLXYyXzAxL2V2YWwiOnRydWUsIldhbGtlcjJkLXYyXzAxL3RyYWluIjpmYWxzZSwiV2Fsa2VyMmQtdjJfMDIvZXZhbCI6dHJ1ZSwiV2Fsa2VyMmQtdjJfMDIvdHJhaW4iOmZhbHNlLCJXYWxrZXIyZC12Ml8wMy9ldmFsIjp0cnVlLCJXYWxrZXIyZC12Ml8wMy90cmFpbiI6ZmFsc2UsIldhbGtlcjJkLXYyXzA0L2V2YWwiOnRydWUsIldhbGtlcjJkLXYyXzA0L3RyYWluIjpmYWxzZX0%3D)
Ant-v2 (Mujoco)         | 4,714@1M      | 4,500@3M      | [tensorboard.dev](https://tensorboard.dev/experiment/4ywD2XzOS2GVJgdkY19c9w/#scalars&runSelectionState=eyJBbnQtdjJfMDAvZXZhbCI6dHJ1ZSwiQW50LXYyXzAwL3RyYWluIjpmYWxzZSwiQW50LXYyXzAxL2V2YWwiOnRydWUsIkFudC12Ml8wMS90cmFpbiI6ZmFsc2UsIkFudC12Ml8wMi9ldmFsIjp0cnVlLCJBbnQtdjJfMDIvdHJhaW4iOmZhbHNlLCJBbnQtdjJfMDMvZXZhbCI6dHJ1ZSwiQW50LXYyXzAzL3RyYWluIjpmYWxzZSwiQW50LXYyXzA0L2V2YWwiOnRydWUsIkFudC12Ml8wNC90cmFpbiI6ZmFsc2V9)
Humanoid-v2 (Mujoco)    | 7,098@5M      | 7,391@10M     | [tensorboard.dev](https://tensorboard.dev/experiment/L9KkrwFzQrKArX0xMjs8Cg/#scalars&runSelectionState=eyJIdW1hbm9pZC12Ml8wMC9ldmFsIjp0cnVlLCJIdW1hbm9pZC12Ml8wMC90cmFpbiI6ZmFsc2UsIkh1bWFub2lkLXYyXzAxL2V2YWwiOnRydWUsIkh1bWFub2lkLXYyXzAxL3RyYWluIjpmYWxzZSwiSHVtYW5vaWQtdjJfMDIvZXZhbCI6dHJ1ZSwiSHVtYW5vaWQtdjJfMDIvdHJhaW4iOmZhbHNlLCJIdW1hbm9pZC12Ml8wMy9ldmFsIjp0cnVlLCJIdW1hbm9pZC12Ml8wMy90cmFpbiI6ZmFsc2UsIkh1bWFub2lkLXYyXzA0L2V2YWwiOnRydWUsIkh1bWFub2lkLXYyXzA0L3RyYWluIjpmYWxzZX0%3D)

The sections that follow contain figures showing the mean along with min/max
ranges for each of 5 runs of the environments tested along with details of the
test environment and commands used. Due to all of the necessary libraries, we
created a
[Docker](https://github.com/tensorflow/agents/tree/master/tools/docker) for
TF-Agents that includes [MuJoCo](http://www.mujoco.org/). Using the Docker and
executing an experiment is detailed in the
[Reproduce results](#reproduce-results) section. The commands should also work
on a bare metal workstation setup as well.

## HalfCheetah-v2 (Mujoco)

![alt_text](https://raw.githubusercontent.com/tensorflow/agents/master/docs/images/sac_readme/halfcheetah-v2_graph.png "HalfCheetah-v2 Mean and min/max graph.")

```shell
$  python3 tf_agents/experimental/examples/sac/haarnoja18/sac_train_eval.py \
     --root_dir=./logs/HalfCheetah-v2_00/ \
     --gin_file=tf_agents/experimental/examples/sac/haarnoja18/configs/half_cheetah.gin \
     --num_iterations=3000000 \
     --alsologtostderr
```

## Hopper-v2 (Mujoco)

![alt_text](https://raw.githubusercontent.com/tensorflow/agents/master/docs/images/sac_readme/hopper-v2_graph.png "Hopper-v2 Mean and min/max graph.")

```shell
$  python3 tf_agents/experimental/examples/sac/haarnoja18/sac_train_eval.py \
     --root_dir=./logs/Hopper-v2_00/ \
     --gin_file=tf_agents/experimental/examples/sac/haarnoja18/configs/hopper.gin \
     --num_iterations=1000000 \
     --alsologtostderr
```

## Walker2d-v2 (Mujoco)

![alt_text](https://raw.githubusercontent.com/tensorflow/agents/master/docs/images/sac_readme/walker2d-v2_graph.png "Walker2d-v2 Mean and min/max graph.")

```shell
$  python3 tf_agents/experimental/examples/sac/haarnoja18/sac_train_eval.py \
     --root_dir=./logs/Walker2d-v2_00/ \
     --gin_file=tf_agents/experimental/examples/sac/haarnoja18/configs/walker_2d.gin \
     --num_iterations=3000000 \
     --alsologtostderr
```

## Ant-v2 (Mujoco)

![alt_text](https://raw.githubusercontent.com/tensorflow/agents/master/docs/images/sac_readme/ant-v2_graph.png "Ant-v2 Mean and min/max graph.")

```shell
$  python3 tf_agents/experimental/examples/sac/haarnoja18/sac_train_eval.py \
     --root_dir=./logs/Ant-v2_00/ \
     --gin_file=tf_agents/experimental/examples/sac/haarnoja18/configs/ant.gin \
     --num_iterations=3000000 \
     --alsologtostderr
```

## Humanoid-v2 (Mujoco)

![alt_text](https://raw.githubusercontent.com/tensorflow/agents/master/docs/images/sac_readme/humanoid-v2_graph.png "Humanoid-v2 Mean and min/max graph.")

```shell
$  python3 tf_agents/experimental/examples/sac/haarnoja18/sac_train_eval.py \
     --root_dir=./logs/Humanoid-v2_00/ \
     --gin_file=tf_agents/experimental/examples/sac/haarnoja18/configs/humanoid.gin \
     --num_iterations=10000000 \
     --alsologtostderr
```

**Deviations from standard testing environment:**

*   The Humanoid experiment ran on a `n1-standard-32` instance for a small
    improvement in total time to run the experiment.

## Reproduce results

To reproduce the numbers we suggest using the commands below, which is using the
head of TF-Agents. Unless specified in the details of section, each experiment
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
$  export INSTANCE_NAME=sac-training

# 16 vCPU
$  gcloud compute instances create $INSTANCE_NAME \
    --project=$PROJECT \
    --zone=$ZONE \
    --machine-type=n1-standard-16 \
    --maintenance-policy=TERMINATE \
    --image=$IMAGE \
    --image-project=ml-images \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd

# It may take a couple minutes for the VM to be available
# This copies your MuJoCo key to the instance from your local machine.
$  gcloud compute scp --zone $ZONE --project $PROJECT $HOME/.mujoco/mjkey.txt $INSTANCE_NAME:/home/$USER/

# Login to the cloud instance.
$  gcloud compute ssh --zone $ZONE --project $PROJECT $INSTANCE_NAME
```

**Step 2:** Build a
[Docker](https://github.com/tensorflow/agents/tree/master/tools/docker) with
MuJoCo to be used for the experiment. These steps take place on the instance
that was created in step 1.

```shell
$  git clone https://github.com/tensorflow/agents.git && cd agents
# Moves MuJoco key scp'd in step 1 into location for docker build.
$  mv ../mjkey.txt .

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
     python3 tf_agents/experimental/examples/sac/haarnoja18/sac_train_eval.py \
       --root_dir=./logs/HalfCheetah-v2_00/ \
       --gin_file=tf_agents/experimental/examples/sac/haarnoja18/configs/half_cheetah.gin \
       -- num_iterations=3000000 \
       --alsologtostderr
```

**Step 4 (Optional):** Use [tensorboard.dev](https://tensorboard.dev/) to track
progress and save the results.

```shell
# To upload the results to tensorboard.dev (public) exit out of the tmux session
# and start another session uploading the results. This command is run outside
# of Docker and assumes tensorboard is installed on the host system.
# ctrl + b followed by d will detach from the session
$  tmux new -s tensorboard
$  tensorboard dev upload --logdir=./logs/
```

**Addendum**

Due to a fix for the Mujoco docker that was not checked in, running the exact
code for the recorded results requires a specific git hash and cherry-pick of
the docker build file. We do not recommend this step because the `head` is
tested nightly. This information is recorded here so we remember exactly how the
data was collected with warts and all.

```shell
# After checking out the code above and before building the docker execute the following:
$  git reset --hard 75a2e7f9bef90d03fbf371321293aba4bacff255
$  git cherry-pick a7aeba2fef8d4908a4a3f7513ccddc887d4c7c32
```

`tf-agents-nightly==0.7.0.dev20200905` and `tf-nightly==2.4.0.dev20200904` were
used to generate the results.
