import time
import pickle
import sys
import gym.spaces
import logz
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch import nn, optim
from collections import namedtuple
from dqn_utils import LinearSchedule, ReplayBuffer, get_wrapper_by_name

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_lambda"])
 

class QLearner(object):

  def __init__(
    self,
    env,
    q_func,
    optimizer_spec,
    exploration=LinearSchedule(1000000, 0.1),
    stopping_criterion=None,
    replay_buffer_size=1000000,
    batch_size=32,
    gamma=0.99,
    learning_starts=50000,
    learning_freq=4,
    frame_history_len=4,
    target_update_freq=10000,
    grad_norm_clipping=10,
    double_q=True,
    lander=False):
    """Run Deep Q-learning algorithm.

    You can specify your own convnet using q_func.

    All schedules are w.r.t. total number of steps taken in the environment.

    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            in_channels: int
                number of channels for the input
            num_actions: int
                number of actions
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    exploration: rl_algs.deepq.utils.schedules.Schedule
        schedule for probability of chosing random action.
    stopping_criterion: (env, t) -> bool
        should return true when it's ok for the RL algorithm to stop.
        takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    grad_norm_clipping: float or None
        If not None gradients' norms are clipped to this value.
    double_q: bool
        If True, then use double Q-learning to compute target values. Otherwise, use vanilla DQN.
        https://papers.nips.cc/paper/3964-double-q-learning.pdf
    """
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete

    self.target_update_freq = target_update_freq
    self.optimizer_spec = optimizer_spec
    self.batch_size = batch_size
    self.learning_freq = learning_freq
    self.learning_starts = learning_starts
    self.stopping_criterion = stopping_criterion
    self.env = env
    self.exploration = exploration
    self.gamma = gamma
    self.double_q = double_q
    self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    
    ###############
    # BUILD MODEL #
    ###############

    if len(self.env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        in_features = self.env.observation_space.shape[0]
    else:
        img_h, img_w, img_c = self.env.observation_space.shape
        in_features = frame_history_len * img_c
    self.num_actions = self.env.action_space.n

    # define deep Q network and target Q network
    self.q_net = q_func(in_features, self.num_actions).to(self.device)
    self.target_q_net = q_func(in_features, self.num_actions).to(self.device)

    # construct optimization op (with gradient clipping)
    parameters = self.q_net.parameters()
    self.optimizer = self.optimizer_spec.constructor(parameters, lr=1, 
                                                     **self.optimizer_spec.kwargs)
    self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, self.optimizer_spec.lr_lambda)
    # clip_grad_norm_fn will be called before doing gradient decent
    self.clip_grad_norm_fn = lambda : nn.utils.clip_grad_norm_(parameters, max_norm=grad_norm_clipping)

    # update_target_fn will be called periodically to copy Q network to target Q network
    self.update_target_fn = lambda : self.target_q_net.load_state_dict(self.q_net.state_dict())

    # construct the replay buffer
    self.replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len, lander=lander)
    self.replay_buffer_idx = None

    ###############
    # RUN ENV     #
    ###############
    self.model_initialized = False
    self.num_param_updates = 0
    self.mean_episode_reward      = -float('nan')
    self.best_mean_episode_reward = -float('inf')
    self.last_obs = self.env.reset()
    self.log_every_n_steps = 10000

    self.start_time = time.time()
    self.t = 0

  def calc_loss(self, obs, ac, rw, nxobs, done):
    """
        Calculate the loss for a batch of transitions. 

        Here, you should fill in your own code to compute the Bellman error. This requires
        evaluating the current and next Q-values and constructing the corresponding error.

        arguments:
            ob: The observation for current step
            ac: The corresponding action for current step
            rw: The reward for each timestep
            nxob: The observation after taking one step forward
            done: The mask for terminal state. This value is 1 if the next state corresponds to
                  the end of an episode, in which case there is no Q-value at the next state;
                  at the end of an episode, only the current state reward contributes to the target,
                  not the next state Q-value (i.e. target is just rew_t_ph, not rew_t_ph + gamma * q_tp1)

            inputs are generated from self.replay_buffer.sample, you can refer the code in dqn_utils.py
            for more details 

        returns:
            a scalar tensor represent the loss

        Hint: use smooth_l1_loss (a.k.a huber_loss) instead of mean squared error.
              use self.double_q to switch between double DQN and vanilla DQN.
    """
    
    # YOUR CODE HERE
    
    
  def stopping_criterion_met(self):
    return self.stopping_criterion is not None and self.stopping_criterion(self.env, self.t)

  def step_env(self):
    ### 2. Step the env and store the transition
    # At this point, "self.last_obs" contains the latest observation that was
    # recorded from the simulator. Here, your code needs to store this
    # observation and its outcome (reward, next observation, etc.) into
    # the replay buffer while stepping the simulator forward one step.
    # At the end of this block of code, the simulator should have been
    # advanced one step, and the replay buffer should contain one more
    # transition.
    # Specifically, self.last_obs must point to the new latest observation.
    # Useful functions you'll need to call:
    # obs, reward, done, info = env.step(action)
    # this steps the environment forward one step
    # obs = env.reset()
    # this resets the environment if you reached an episode boundary.
    # Don't forget to call env.reset() to get a new observation if done
    # is true!!
    # Note that you cannot use "self.last_obs" directly as input
    # into your network, since it needs to be processed to include context
    # from previous frames. You should check out the replay buffer
    # implementation in dqn_utils.py to see what functionality the replay
    # buffer exposes. The replay buffer has a function called
    # encode_recent_observation that will take the latest observation
    # that you pushed into the buffer and compute the corresponding
    # input that should be given to a Q network by appending some
    # previous frames.
    # Don't forget to include epsilon greedy exploration!
    # And remember that the first time you enter this loop, the model
    # may not yet have been initialized (but of course, the first step
    # might as well be random, since you haven't trained your net...)

    #####

    # YOUR CODE HERE
    

  def update_model(self):
    ### 3. Perform experience replay and train the network.
    # note that this is only done if the replay buffer contains enough samples
    # for us to learn something useful -- until then, the model will not be
    # initialized and random actions should be taken
    self.lr_scheduler.step()
    
    if (self.t > self.learning_starts and \
        self.t % self.learning_freq == 0 and \
        self.replay_buffer.can_sample(self.batch_size)):
      
      # Here, you should perform training. Training consists of four steps:
      # 3.a: use the replay buffer to sample a batch of transitions (see the
      # replay buffer code for function definition, each batch that you sample
      # should consist of current observations, current actions, rewards,
      # next observations, and done indicator).
      # 3.b: set the self.model_initialized to True. Because the newwork in starting
      # to train, and you will use it to take action in self.step_env.
      # 3.c: train the model. To do this, you'll need to use the self.optimizer and
      # self.calc_loss that were created earlier: self.calc_loss is what you
      # created to compute the total Bellman error in a batch, and self.optimizer
      # will actually perform a gradient step and update the network parameters
      # to reduce the loss. 
      # Before your optimizer take step, don`t forget to call self.clip_grad_norm_fn
      # to perform gradient clipping.
      # 3.d: periodically update the target network by calling self.update_target_fn
      # you should update every target_update_freq steps, and you may find the
      # variable self.num_param_updates useful for this (it was initialized to 0)
      #####
      
      # YOUR CODE HERE
      

      self.num_param_updates += 1

    self.t += 1

  def log_progress(self):
    episode_rewards = get_wrapper_by_name(self.env, "Monitor").get_episode_rewards()

    if len(episode_rewards) > 0:
      self.mean_episode_reward = np.mean(episode_rewards[-100:])

    if len(episode_rewards) > 100:
      self.best_mean_episode_reward = max(self.best_mean_episode_reward, self.mean_episode_reward)

    if self.t % self.log_every_n_steps == 0 and self.model_initialized:
      logz.log_tabular("TimeStep", self.t)
      logz.log_tabular("MeanReturn", self.mean_episode_reward)
      logz.log_tabular("BestMeanReturn", max(self.best_mean_episode_reward, self.mean_episode_reward))
      logz.log_tabular("Episodes", len(episode_rewards))
      logz.log_tabular("Exploration", self.exploration.value(self.t))
      logz.log_tabular("LearningRate", self.optimizer_spec.lr_lambda(self.t))
      logz.log_tabular("Time", (time.time() - self.start_time) / 60.)
      logz.dump_tabular()
      logz.save_pytorch_model(self.q_net)
      
def learn(*args, **kwargs):
  alg = QLearner(*args, **kwargs)
  while not alg.stopping_criterion_met():
    alg.step_env()
    # at this point, the environment should have been advanced one step (and
    # reset if done was true), and self.last_obs should point to the new latest
    # observation
    alg.update_model()
    alg.log_progress()

