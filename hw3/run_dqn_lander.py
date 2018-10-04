import gym
from gym import wrappers
import time
import logz
import os.path as osp
import random
import numpy as np
import torch
from torch import nn

import dqn
from dqn_utils import ConstantSchedule, PiecewiseSchedule, get_wrapper_by_name


def weights_init(m):
    if hasattr(m, 'weight'):
        nn.init.orthogonal_(m.weight)
    if hasattr(m, 'bias'):
        nn.init.constant_(m.bias, 0)

class DQN(nn.Module): # for lunar lander
    def __init__(self, in_features, num_actions):
        super(DQN, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, out_features=64),
            nn.ReLU(True),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(True),
            nn.Linear(in_features=64, out_features=num_actions),
        )

        self.apply(weights_init)

    def forward(self, obs):
        out = self.classifier(obs)
        return out

def lander_optimizer():
    lr_schedule = ConstantSchedule(1e-3)
    lr_lambda = lambda t: lr_schedule.value(t)
    return dqn.OptimizerSpec(
        constructor=torch.optim.Adam,
        lr_lambda=lr_lambda,
        kwargs={}
    )

def lander_stopping_criterion(num_timesteps):
    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps
    return stopping_criterion

def lander_exploration_schedule(num_timesteps):
    return PiecewiseSchedule(
        [
            (0, 1),
            (num_timesteps * 0.1, 0.02),
        ], outside_value=0.02
    )

def lander_kwargs():
    return {
        'optimizer_spec': lander_optimizer(),
        'q_func': DQN,
        'replay_buffer_size': 50000,
        'batch_size': 32,
        'gamma': 1.00,
        'learning_starts': 1000,
        'learning_freq': 1,
        'frame_history_len': 1,
        'target_update_freq': 3000,
        'grad_norm_clipping': 10,
        'lander': True
    }

def lander_learn(env,
                 num_timesteps):

    optimizer = lander_optimizer()
    stopping_criterion = lander_stopping_criterion(num_timesteps)
    exploration_schedule = lander_exploration_schedule(num_timesteps)

    dqn.learn(
        env=env,
        exploration=lander_exploration_schedule(num_timesteps),
        stopping_criterion=lander_stopping_criterion(num_timesteps),
        double_q=True,
        **lander_kwargs()
    )
    env.close()

def set_global_seeds(i):
    torch.manual_seed(i)
    if torch.cuda.is_available:
        torch.cuda.manual_seed(i)
    np.random.seed(i)
    random.seed(i)

def get_env(env_name, exp_name, seed):
    env = gym.make(env_name)

    set_global_seeds(seed)
    env.seed(seed)

    # Set Up Logger
    logdir = 'dqn_' + exp_name + '_' + env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = osp.join('data', logdir)
    logdir = osp.join(logdir, '%d'%seed)
    logz.configure_output_dir(logdir)
    hyperparams = {'exp_name': exp_name, 'env_name': env_name}
    logz.save_hyperparams(hyperparams)

    expt_dir = '/tmp/hw3_vid_dir/'
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True, video_callable=False)
    

    return env

def main():
    # Choose Atari games.
    env_name = 'LunarLander-v2'
    exp_name = 'LunarLander_double_dqn' # you can use it to mark different experiments
    
    # Run training
    seed = 4565 # you may want to randomize this
    print('random seed = %d' % seed)
    env = get_env(env_name, exp_name, seed)
    lander_learn(env, num_timesteps=500000)

if __name__ == "__main__":
    main()
