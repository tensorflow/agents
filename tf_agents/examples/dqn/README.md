# DQNAgent

DQNAgent implements the deep Q-network (DQN) algorithm algorithm from
["Human level control through deep reinforcement learning" ](https://deepmind.com/research/dqn/)
(Mnih et al., 2015).

# Examples

Here is an example of training and evaluating DQN in the Pong-v0 environment:

```shell
$  python tf_agents/experimental/examples/dqn/mnih15/dqn_train_eval_atari.py  \
     --num_iterations=12500000 \
     --root_dir=$HOME/tmp/dqn/pong_v0/

# To view progress in TensorBoard use another terminal and execute.
$  tensorboard --logdir $HOME/tmp/dqn/pong_v0/
```

Note: [Issue #475](https://github.com/tensorflow/agents/issues/475):
The default setup does not checkpoint the policy or data. If the pipeline gets
preempted, training and evaluation will start over. You may observe more than
one learning curve in this situation. If you enable policy checkpointing, the
learner will reload the most recent saved policy. However, because we are not
yet checkpointing the data from Reverb, we refill the table with the data
collected by the current policy. We lose the data collected with older policies.
This will result in distortions in the return that we will address in the future.

# Validation and Performance

To validate TF-Agents’ DQN implementation, we ran DQN on Pong-v0 using the
hyperparameters from "Human level control through deep reinforcement learning"
(Mnih et al., 2015). The table below is a summary of the results listing the
mean of 5 experiments after 12.5M env steps which amounts to 50M frames
(4 frames per env step). The graph that follows is the mean with the min/max
range.

Env                     | Measurement 1 | Tensorboard
----------------------- | ------------- | -----------
Pong-v0 (Atari)         | 18.8@50M      | [tensorboard.dev](https://tensorboard.dev/experiment/pIkiBSxqQvecWWARJ1Y8uA/#scalars&runSelectionState=eyJkcW5fcG9uZ18wMi90cmFpbiI6ZmFsc2UsImRxbl9wb25nXzAxL3RyYWluIjpmYWxzZSwiZHFuX3BvbmdfMDAvdHJhaW4iOmZhbHNlLCJkcW5fcG9uZ18wMy90cmFpbiI6ZmFsc2UsImRxbl9wb25nXzA0L3RyYWluIjpmYWxzZX0%3D)


![alt_text](https://raw.githubusercontent.com/tensorflow/agents/master/docs/images/dqn_readme/pong-v0_graph.png "Pong-v0 Mean and min/max graph.")

## Reproduce results

To reproduce the numbers we suggest using the commands below, which is using the
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
$  export INSTANCE_NAME=dqn-training

# 32 vCPU
$  gcloud compute instances create $INSTANCE_NAME \
    --project=$PROJECT \
    --zone=$ZONE \
    --machine-type=n1-standard-32 \
    --maintenance-policy=TERMINATE \
    --image=$IMAGE \
    --image-project=ml-images \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd
    --scopes=https://www.googleapis.com/auth/cloud-platform

# Login to the cloud instance.
$  gcloud compute ssh --zone $ZONE --project $PROJECT $INSTANCE_NAME
```

**Step 2:** Build a
[Docker](https://github.com/tensorflow/agents/tree/master/tools/docker) to be
used for the experiment. These steps take place on the instance that was
created in step 1.

```shell
$  git clone https://github.com/tensorflow/agents.git && cd agents

# Core tf-agents docker.
$  docker build -t tf_agents/core \
     --build-arg tf_agents_pip_spec=tf-agents-nightly[reverb] \
     - < tools/docker/ubuntu_1804_tf_agents

```

**Step 3:** Create a tmux session and start the train and eval.

```shell
# Using tmux keeps the experiment running if the connection drops.
$  tmux new -s bench

# Runs DQN on Pong in the docker instance.
$  docker run --rm -it -v $(pwd):/workspace -w /workspace/ tf_agents/core \
     python3 tf_agents/experimental/examples/dqn/mnih15/dqn_train_eval_atari.py  \
       --num_iterations=12500000 \
       --root_dir=./logs/dqn_pong_01/
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

Below is the git hash and version of TensorFlow used to get the results above.
This is here as an audit trail.

```shell
# After checking out the code above and before building the docker execute the following:
$  git reset --hard 55fb34fabba14512e9b090743277764129573fec
```

`tf-nightly==2.5.0.dev20201215` was used to generate the results.



