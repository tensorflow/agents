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
$  python tf_agents/examples/sac/haarnoja18/sac_train_eval.py \
     --root_dir=$HOME/tmp/sac/HalfCheetah-v2/ \
     --gin_file=tf_agents/examples/sac/haarnoja18/configs/half_cheetah.gin \
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
[configs](https://github.com/tensorflow/agents/tree/master/tf_agents/examples/sac/haarnoja18/configs)
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
HalfCheetah-v2 (Mujoco) | 12,096@1M     | 14,877@3M     | [tensorboard.dev](https://tensorboard.dev/experiment/MD1SVkFARDKnwoqirHJ7Iw/#scalars&runSelectionState=eyJIYWxmQ2hlZXRhaC12Ml8wMC90cmFpbiI6ZmFsc2UsIkhhbGZDaGVldGFoLXYyXzAxL3RyYWluIjpmYWxzZSwiSGFsZkNoZWV0YWgtdjJfMDIvdHJhaW4iOmZhbHNlLCJIYWxmQ2hlZXRhaC12Ml8wMy90cmFpbiI6ZmFsc2UsIkhhbGZDaGVldGFoLXYyXzA0L3RyYWluIjpmYWxzZX0%3D)
Hopper-v2 (Mujoco)      | 3,323@1M      | n/a           | [tensorboard.dev](https://tensorboard.dev/experiment/Q3q2E5zXQmCVEFlS0EAHGA/#scalars&runSelectionState=eyJIb3BwZXItdjJfMDAvdHJhaW4iOmZhbHNlLCJIb3BwZXItdjJfMDEvdHJhaW4iOmZhbHNlLCJIb3BwZXItdjJfMDIvdHJhaW4iOmZhbHNlLCJIb3BwZXItdjJfMDMvdHJhaW4iOmZhbHNlLCJIb3BwZXItdjJfMDQvdHJhaW4iOmZhbHNlfQ%3D%3D)
Walker2d-v2 (Mujoco)    | 4,966@1M      | 5,612@3M      | [tensorboard.dev](https://tensorboard.dev/experiment/I3nq1OQ7QiqWXexD1KVaTg/#scalars&runSelectionState=eyJXYWxrZXIyZC12Ml8wNC9ldmFsIjp0cnVlLCJXYWxrZXIyZC12Ml8wNC90cmFpbiI6ZmFsc2UsIldhbGtlcjJkLXYyXzAyL3RyYWluIjpmYWxzZSwiV2Fsa2VyMmQtdjJfMDIvZXZhbCI6dHJ1ZSwiV2Fsa2VyMmQtdjJfMDEvdHJhaW4iOmZhbHNlLCJXYWxrZXIyZC12Ml8wMS9ldmFsIjp0cnVlLCJXYWxrZXIyZC12Ml8wMC90cmFpbiI6ZmFsc2UsIldhbGtlcjJkLXYyXzAwL2V2YWwiOnRydWUsIldhbGtlcjJkLXYyXzAzL2V2YWwiOnRydWUsIldhbGtlcjJkLXYyXzAzL3RyYWluIjpmYWxzZX0%3D)
Ant-v2 (Mujoco)         | 5,494@1M      | 5,561@3M      | [tensorboard.dev](https://tensorboard.dev/experiment/jsbkOQeERTmBOy0XoCMQcw/#scalars&runSelectionState=eyJBbnQtdjJfMDAvdHJhaW4iOmZhbHNlLCJBbnQtdjJfMDEvdHJhaW4iOmZhbHNlLCJBbnQtdjJfMDIvdHJhaW4iOmZhbHNlLCJBbnQtdjJfMDMvdHJhaW4iOmZhbHNlLCJBbnQtdjJfMDQvdHJhaW4iOmZhbHNlfQ%3D%3D)
Humanoid-v2 (Mujoco)    | 7,455@5M      | 8,114@10M     | [tensorboard.dev](https://tensorboard.dev/experiment/Xk9qg2cmRc2R2Mzr18LXgw/#scalars&runSelectionState=eyJIdW1hbm9pZC12Ml8wNC90cmFpbiI6ZmFsc2UsIkh1bWFub2lkLXYyXzAzL3RyYWluIjpmYWxzZSwiSHVtYW5vaWQtdjJfMDIvdHJhaW4iOmZhbHNlLCJIdW1hbm9pZC12Ml8wMS90cmFpbiI6ZmFsc2UsIkh1bWFub2lkLXYyXzAwL3RyYWluIjpmYWxzZX0%3D)

The sections that follow contain figures showing the mean along with min/max
ranges for each of 5 runs of the environments tested along with details of the
test environment and commands used. Due to all of the necessary libraries, we
created a
[Docker](https://github.com/tensorflow/agents/tree/master/tools/docker) for
TF-Agents that includes [MuJoCo](http://www.mujoco.org/). Using the Docker and
executing an experiment is detailed in the
[Reproduce results](#reproduce-results) section. The commands should also work
on a bare metal workstation setup as well.

## HalfCheetah-v2 (MuJoCo)

![alt_text](https://raw.githubusercontent.com/tensorflow/agents/master/docs/images/sac_readme/halfcheetah-v2_graph.png "HalfCheetah-v2 Mean and min/max graph.")

```shell
$  python3 tf_agents/examples/sac/haarnoja18/sac_train_eval.py \
     --root_dir=./logs/HalfCheetah-v2_00/ \
     --gin_file=tf_agents/examples/sac/haarnoja18/configs/half_cheetah.gin \
     --num_iterations=3000000 \
     --alsologtostderr
```

## Hopper-v2 (MuJoCo)

![alt_text](https://raw.githubusercontent.com/tensorflow/agents/master/docs/images/sac_readme/hopper-v2_graph.png "Hopper-v2 Mean and min/max graph.")

```shell
$  python3 tf_agents/examples/sac/haarnoja18/sac_train_eval.py \
     --root_dir=./logs/Hopper-v2_00/ \
     --gin_file=tf_agents/examples/sac/haarnoja18/configs/hopper.gin \
     --num_iterations=1000000 \
     --alsologtostderr
```

## Walker2d-v2 (MuJoCo)

![alt_text](https://raw.githubusercontent.com/tensorflow/agents/master/docs/images/sac_readme/walker2d-v2_graph.png "Walker2d-v2 Mean and min/max graph.")

```shell
$  python3 tf_agents/examples/sac/haarnoja18/sac_train_eval.py \
     --root_dir=./logs/Walker2d-v2_00/ \
     --gin_file=tf_agents/examples/sac/haarnoja18/configs/walker_2d.gin \
     --num_iterations=3000000 \
     --alsologtostderr
```

## Ant-v2 (MuJoCo)

![alt_text](https://raw.githubusercontent.com/tensorflow/agents/master/docs/images/sac_readme/ant-v2_graph.png "Ant-v2 Mean and min/max graph.")

```shell
$  python3 tf_agents/examples/sac/haarnoja18/sac_train_eval.py \
     --root_dir=./logs/Ant-v2_00/ \
     --gin_file=tf_agents/examples/sac/haarnoja18/configs/ant.gin \
     --num_iterations=3000000 \
     --alsologtostderr
```

## Humanoid-v2 (MuJoCo)

![alt_text](https://raw.githubusercontent.com/tensorflow/agents/master/docs/images/sac_readme/humanoid-v2_graph.png "Humanoid-v2 Mean and min/max graph.")

```shell
$  python3 tf_agents/examples/sac/haarnoja18/sac_train_eval.py \
     --root_dir=./logs/Humanoid-v2_00/ \
     --gin_file=tf_agents/examples/sac/haarnoja18/configs/humanoid.gin \
     --num_iterations=10000000 \
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
     python3 tf_agents/examples/sac/haarnoja18/sac_train_eval.py \
       --root_dir=./logs/HalfCheetah-v2_00/ \
       --gin_file=tf_agents/examples/sac/haarnoja18/configs/half_cheetah.gin \
       --num_iterations=3000000 \
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

Below is the githash and version of TensorFlow used to get the results above.
This is here for an audit trail.

```shell
# After checking out the code above and before building the docker execute the following:
$  git reset --hard 0c82138c4e36dc851c0ed97067cf3118ee9c07c1
```

`tf-nightly==2.4.0.dev20200904` was used to generate the results.
