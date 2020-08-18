# Overview

SACAgent implements the Soft Actor-Critic (SAC) algorithm from ["Soft Actor-Critic Algorithms and Applications" ](https://arxiv.org/pdf/1812.05905.pdf)(Haarnoja, 18). SAC is an off-policy actor-critic deep RL algorithm. Its policy is trained to maximize the combination of return and entropy to encourage exploration. SAC was shown to achieve high stability and sample efficiency in various benchmarks. It is also able to learn complex tasks such as 21-dimensional Humanoid.

SACAgent supports custom environments with continuous action spaces. Both the actor and the critic networks support RNNs.

SACAgent also supports distributed training on multiple GPUs or TPUs when used with the Actor-Learner API. We have a tutorial on distributed training coming soon.

# Examples

Here is a simple example to train and evaluate SACAgent in the HalfCheetah environment:

```
tensorboard --logdir $HOME/tmp/sac/HalfCheetah-v2/ --port 2223 &

python tf_agents/experimental/examples/sac/haarnoja18/sac_train_eval.py \
--root_dir=$HOME/tmp/sac/HalfCheetah-v2/ \
--gin_file=tf_agents/experimental/examples/sac/haarnoja18/configs/half_cheetah.gin \
-- num_iterations=3000000 \
--alsologtostderr
```

We set default hyperparameters according to the SAC paper. We made minor modifications in the number of initial collect steps, reward scaling (0.1 instead of 1.0) and the heuristic entropy target (-dim(A)/2 instead of -dim(A)). Those modifications provides slight benefit in more difficult gym-mujoco environments. Note that optimal reward scaling is likely to be different for other environment suites. We provide gin files to config train_eval hyperparameter arguments for several gym-mujoco environments used in the SAC paper in the `configs` directory.

TODO(b/165023684): Checkpoint the data and reload in the case of preemption.
Note that the default setup does not checkpoint the policy or the data. If the pipeline gets preempted, training and evaluation will start over. You may observe more than one learning curves in this situation. If you enable policy checkpointing, the learner will reload the most recent saved policy. However, because we are not yet checkpointing the data from Reverb, we refill the table with the data collected by the current policy. We lose the data collected with older policies. This will result in distortions in the return that we will address in the future.

# Validation and Performance
We validated the performance on all Mujoco environments reported in the paper, including HalfCheetah-v2, Walker2d-v2, Hopper-v2, Ant-v2 and Humanoid-v2. The experiments were done by launching `sac_train_eval.py`. We confirmed that using the same network architecture and hyperparameters, SACAgent achieves similar performances as the paper, after taking the reported number of steps.

TODO(b/158509522): Add the return curves by the end of August, 2020.

We also monitor the performance nightly in order to catch potential regressions promptly.
