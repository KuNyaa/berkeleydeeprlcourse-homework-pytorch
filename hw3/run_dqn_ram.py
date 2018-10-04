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
from dqn_utils import PiecewiseSchedule, get_wrapper_by_name
from atari_wrappers import wrap_deepmind_ram

def weights_init(m):
    if hasattr(m, 'weight'):
        nn.init.xavier_uniform_(m.weight)
    if hasattr(m, 'bias'):
        nn.init.constant_(m.bias, 0)
        
class DQN(nn.Module): # for atari ram
    def __init__(self, in_features, num_actions):
        super(DQN, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, out_features=256),
            nn.ReLU(True),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(True),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(True),
            nn.Linear(in_features=64, out_features=num_actions),
        )

        self.apply(weights_init)

    def forward(self, obs):
        out = obs.float() / 255 # convert 8-bits ram state to float in [0, 1]
        out = self.classifier(out)
        return out

def atari_learn(env,
                num_timesteps):
    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0

    lr_multiplier = 1.0
    lr_schedule = PiecewiseSchedule(
        [
            (0,                   1e-4 * lr_multiplier),
            (num_iterations / 10, 1e-4 * lr_multiplier),
            (num_iterations / 2,  5e-5 * lr_multiplier),
        ],
        outside_value=5e-5 * lr_multiplier
    )
    lr_lambda = lambda t: lr_schedule.value(t)

    optimizer = dqn.OptimizerSpec(
        constructor=torch.optim.Adam,
        kwargs=dict(eps=1e-4),
        lr_lambda=lr_lambda
    )

    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 0.2),
            (1e6, 0.1),
            (num_iterations / 2, 0.01),
        ], outside_value=0.01
    )

    dqn.learn(
        env,
        q_func=DQN,
        optimizer_spec=optimizer,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=1000000,
        batch_size=32,
        gamma=0.99,
        learning_starts=50000,
        learning_freq=4,
        frame_history_len=1,
        target_update_freq=10000,
        grad_norm_clipping=10
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
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)
    env = wrap_deepmind_ram(env)

    return env

def main():
    # Choose Atari games.
    env_name = 'Pong-ram-v0'
    exp_name = 'Pong_double_dqn' # you can use it to mark different experiments
    
    # Run training
    seed = 0 # Use a seed of zero (you may want to randomize the seed!)
    print('random seed = %d' % seed)
    env = get_env(env_name, exp_name, seed)
    atari_learn(env, num_timesteps=int(4e7))

if __name__ == "__main__":
    main()
