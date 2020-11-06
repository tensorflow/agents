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
$  python tf_agents/experimental/examples/ppo/schulman17/ppo_clip_train_eval.py \
     --root_dir=$HOME/tmp/ppo/HalfCheetah-v2/ \
     --gin_file=tf_agents/experimental/examples/ppo/schulman17/configs/half_cheetah.gin \
     --alsologtostderr

# To view progress in TensorBoard use another terminal and execute.
$  tensorboard --logdir $HOME/tmp/ppo/HalfCheetah-v2/
```

We set default hyperparameters according to the PPO paper. We provide gin files
to config train_eval hyperparameter arguments for several gym-mujoco
environments in the [configs](https://github.com/tensorflow/agents/tree/master/tf_agents/experimental/examples/ppo/schulman17/configs)
directory.

