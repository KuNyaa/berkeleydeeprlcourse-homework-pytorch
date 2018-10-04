"""
Original code from John Schulman for CS294 Deep Reinforcement Learning Spring 2017
Adapted for CS294-112 Fall 2017 by Abhishek Gupta and Joshua Achiam
Adapted for CS294-112 Fall 2018 by Soroush Nasiriany, Sid Reddy, and Greg Kahn
Adapted for pytorch version by Ning Dai
"""
import numpy as np
import torch
import gym
import logz
import os
import time
import inspect
from torch.multiprocessing import Process
from torch import nn, optim

#============================================================================================#
# Utilities
#============================================================================================#

def build_mlp(input_size, output_size, n_layers, hidden_size, activation=nn.Tanh):
    """
        Builds a feedforward neural network
        
        arguments:
            input_size: size of the input layer
            output_size: size of the output layer
            n_layers: number of hidden layers
            hidden_size: dimension of the hidden layers
            activation: activation of the hidden layers
            output_activation: activation of the output layer

        returns:
            an instance of nn.Sequential which contains the feedforward neural network

        Hint: use nn.Linear
    """
    layers = []
    # YOUR HW2 CODE HERE
    raise NotImplementedError

    return nn.Sequential(*layers).apply(weights_init)

def weights_init(m):
    if hasattr(m, 'weight'):
        nn.init.xavier_uniform_(m.weight)

def pathlength(path):
    return len(path["reward"])

def setup_logger(logdir, locals_):
    # Configure output directory for logging
    logz.configure_output_dir(logdir)
    # Log experimental parameters
    args = inspect.getargspec(train_AC)[0]
    hyperparams = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_hyperparams(hyperparams)

class PolicyNet(nn.Module):
    def __init__(self, neural_network_args):
        super(PolicyNet, self).__init__()
        self.ob_dim = neural_network_args['ob_dim']
        self.ac_dim = neural_network_args['ac_dim']
        self.discrete = neural_network_args['discrete']
        self.hidden_size = neural_network_args['size']
        self.n_layers = neural_network_args['actor_n_layers']

        self.define_model_components()
        
    def define_model_components(self):
        """
            Define the parameters of policy network here.
            You can use any instance of nn.Module or nn.Parameter.

            Hint: use the 'build_mlp' function above
                In the discrete case, model should output logits of a categorical distribution
                    over the actions
                In the continuous case, model should output a tuple (mean, log_std) of a Gaussian
                    distribution over actions. log_std should just be a trainable
                    variable, not a network output.
        """
        # YOUR HW2 CODE HERE
        if self.discrete:
            raise NotImplementedError
        else:
            raise NotImplementedError
            
    #========================================================================================#
    #                           ----------PROBLEM 2----------
    #========================================================================================#
    """
        Notes on notation:
        
        Pytorch tensor variables have the prefix ts_, to distinguish them from the numpy array
        variables that are computed later in the function
    
        Prefixes and suffixes:
        ob - observation 
        ac - action
        _no - this tensor should have shape (batch size, observation dim)
        _na - this tensor should have shape (batch size, action dim)
        _n  - this tensor should have shape (batch size)
            
        Note: batch size is defined at runtime
    """
    def forward(self, ts_ob_no):
        """
            Define forward pass for policy network.

            arguments:
                ts_ob_no: (batch_size, self.ob_dim) 

            returns:
                the parameters of the policy.

                if discrete, the parameters are the logits of a categorical distribution
                    over the actions
                    ts_logits_na: (batch_size, self.ac_dim)

                if continuous, the parameters are a tuple (mean, log_std) of a Gaussian
                    distribution over actions. log_std should just be a trainable
                    variable, not a network output.
                    ts_mean: (batch_size, self.ac_dim)
                    st_logstd: (self.ac_dim,)
        
            Hint: use the components you defined in self.define_model_components
        """
        raise NotImplementedError
        if self.discrete:
            # YOUR HW2 CODE HERE
            ts_logits_na = None
            return ts_logits_na
        else:
            # YOUR HW2 CODE HERE
            ts_mean = None
            ts_logstd = None
            return (ts_mean, ts_logstd)
    
#============================================================================================#
# Actor Critic
#============================================================================================#

class Agent(object):
    def __init__(self, neural_network_args, sample_trajectory_args, estimate_advantage_args):
        super(Agent, self).__init__()
        self.ob_dim = neural_network_args['ob_dim']
        self.ac_dim = neural_network_args['ac_dim']
        self.discrete = neural_network_args['discrete']
        self.hidden_size = neural_network_args['size']
        self.critic_n_layers = neural_network_args['critic_n_layers']
        self.actor_learning_rate = neural_network_args['actor_learning_rate']
        self.critic_learning_rate = neural_network_args['critic_learning_rate']
        self.num_target_updates = neural_network_args['num_target_updates']
        self.num_grad_steps_per_target_update = neural_network_args['num_grad_steps_per_target_update']

        self.animate = sample_trajectory_args['animate']
        self.max_path_length = sample_trajectory_args['max_path_length']
        self.min_timesteps_per_batch = sample_trajectory_args['min_timesteps_per_batch']

        self.gamma = estimate_advantage_args['gamma']
        self.normalize_advantages = estimate_advantage_args['normalize_advantages']

        self.policy_net = PolicyNet(neural_network_args)
        self.value_net = build_mlp(self.ob_dim, 1, self.critic_n_layers, self.hidden_size)

        self.actor_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.value_net.parameters(), lr=self.critic_learning_rate)
        
    def sample_action(self, ob_no):
        """
            Build the method used for sampling action from the policy distribution
    
            arguments:
                ob_no: (batch_size, self.ob_dim)

            returns:
                sampled_ac: 
                    if discrete: (batch_size)
                    if continuous: (batch_size, self.ac_dim)

            Hint: for the continuous case, use the reparameterization trick:
                 The output from a Gaussian distribution with mean 'mu' and std 'sigma' is
        
                      mu + sigma * z,         z ~ N(0, I)
        
                 This reduces the problem to just sampling z. (Hint: use torch.normal!)
        """
        ts_ob_no = torch.from_numpy(ob_no).float()
        
        raise NotImplementedError
        if self.discrete:
            ts_logits_na = self.policy_net(ts_ob_no)
            # YOUR HW2 CODE HERE
            ts_probs = None
            ts_sampled_ac = None
        else:
            ts_mean, ts_logstd = self.policy_net(ts_ob_no)
            # YOUR HW2 CODE HERE
            ts_sampled_ac = None

        sampled_ac = ts_sampled_ac.numpy()
            
        return sampled_ac
    
    def get_log_prob(self, policy_parameters, ts_ac_na):
        """
            Build the method used for computing the log probability of a set of actions
            that were actually taken according to the policy

            arguments:
                policy_parameters
                    if discrete: logits of a categorical distribution over actions 
                        ts_logits_na: (batch_size, self.ac_dim)
                    if continuous: (mean, log_std) of a Gaussian distribution over actions
                        ts_mean: (batch_size, self.ac_dim)
                        ts_logstd: (self.ac_dim,)

                ts_ac_na: (batch_size, self.ac_dim)

            returns:
                ts_logprob_n: (batch_size)

            Hint:
                For the discrete case, use the log probability under a categorical distribution.
                For the continuous case, use the log probability under a multivariate gaussian.
        """
        raise NotImplementedError
        if self.discrete:
            ts_logits_na = policy_parameters
            # YOUR HW2 CODE HERE
            ts_logprob_n = None
        else:
            ts_mean, ts_logstd = policy_parameters
            # YOUR HW2 CODE HERE
            ts_logprob_n = None
            
        return ts_logprob_n

    def sample_trajectories(self, itr, env):
        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            animate_this_episode=(len(paths)==0 and (itr % 10 == 0) and self.animate)
            path = self.sample_trajectory(env, animate_this_episode)
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > self.min_timesteps_per_batch:
                break
        return paths, timesteps_this_batch

    def sample_trajectory(self, env, animate_this_episode):
        ob = env.reset()
        obs, acs, rewards, next_obs, terminals = [], [], [], [], []
        steps = 0
        while True:
            if animate_this_episode:
                env.render()
                time.sleep(0.1)
            obs.append(ob)
            raise NotImplementedError
            ac = None # YOUR HW2 CODE HERE
            ac = ac[0]
            acs.append(ac)
            ob, rew, done, _ = env.step(ac)
            # add the observation after taking a step to next_obs
            # YOUR CODE HERE
            raise NotImplementedError
            rewards.append(rew)
            steps += 1
            # If the episode ended, the corresponding terminal value is 1
            # otherwise, it is 0
            # YOUR CODE HERE
            if done or steps > self.max_path_length:
                raise NotImplementedError
                break
            else:
                raise NotImplementedError
        path = {"observation" : np.array(obs, dtype=np.float32), 
                "reward" : np.array(rewards, dtype=np.float32), 
                "action" : np.array(acs, dtype=np.float32),
                "next_observation": np.array(next_obs, dtype=np.float32),
                "terminal": np.array(terminals, dtype=np.float32)}
        return path

    def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):
        """
            Estimates the advantage function value for each timestep.

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from 
                Agent.sample_trajectories

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                re_n: length: sum_of_path_lengths. Each element in re_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end

            returns:
                adv_n: shape: (sum_of_path_lengths). A single vector for the estimated 
                    advantages whose length is the sum of the lengths of the paths
        """
        # First, estimate the Q value as Q(s, a) = r(s, a) + gamma*V(s')
        # To get the advantage, subtract the V(s) to get A(s, a) = Q(s, a) - V(s)
        # This requires calling the critic twice --- to obtain V(s') when calculating Q(s, a),
        # and V(s) when subtracting the baseline
        # Note: don't forget to use terminal_n to cut off the V(s') term when computing Q(s, a)
        # otherwise the values will grow without bound.
        # YOUR CODE HERE
        raise NotImplementedError
        adv_n = None
        
        if self.normalize_advantages:
            raise NotImplementedError
            adv_n = None # YOUR HW2 CODE HERE
        return adv_n

    def update_critic(self, ob_no, next_ob_no, re_n, terminal_n):
        """
            Update the parameters of the critic.

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                re_n: length: sum_of_path_lengths. Each element in re_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end

            returns:
                nothing
        """
        # Use a bootstrapped target values to update the critic
        # Compute the target values r(s, a) + gamma*V(s') by calling the critic to compute V(s')
        # In total, take n=self.num_grad_steps_per_target_update*self.num_target_updates gradient update steps
        # Every self.num_grad_steps_per_target_update steps, recompute the target values
        # by evaluating V(s') on the updated critic
        # Note: don't forget to use terminal_n to cut off the V(s') term when computing the target
        # otherwise the values will grow without bound.
        # YOUR CODE HERE
        raise NotImplementedError
                
    def update_actor(self, ob_no, ac_na, adv_n):
        """ 
            Update the parameters of the policy.

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                ac_na: shape: (sum_of_path_lengths).
                adv_n: shape: (sum_of_path_lengths). A single vector for the estimated
                    advantages whose length is the sum of the lengths of the paths

            returns:
                nothing

        """
        # convert numpy array to pytorch tensor
        ts_ob_no, ts_ac_na, ts_adv_n = map(lambda x: torch.from_numpy(x), [ob_no, ac_na, adv_n])

        # The policy takes in an observation and produces a distribution over the action space
        policy_parameters = self.policy_net(ts_ob_no)

        # We can compute the logprob of the actions that were actually taken by the policy
        # This is used in the loss function.
        ts_logprob_n = self.get_log_prob(policy_parameters, ts_ac_na)

        # clean the gradient for model parameters
        self.actor_optimizer.zero_grad()
        
        actor_loss = - (ts_logprob_n * ts_adv_n).mean() 
        actor_loss.backward()
        
        self.actor_optimizer.step()

def train_AC(
        exp_name,
        env_name,
        n_iter, 
        gamma, 
        min_timesteps_per_batch, 
        max_path_length,
        actor_learning_rate,
        critic_learning_rate,
        num_target_updates,
        num_grad_steps_per_target_update,
        animate, 
        logdir, 
        normalize_advantages,
        seed,
        actor_n_layers,
        critic_n_layers,
        size):

    start = time.time()

    #========================================================================================#
    # Set Up Logger
    #========================================================================================#
    setup_logger(logdir, locals())

    #========================================================================================#
    # Set Up Env
    #========================================================================================#

    # Make the gym environment
    env = gym.make(env_name)

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)

    # Maximum length for episodes
    max_path_length = max_path_length or env.spec.max_episode_steps

    # Is this env continuous, or self.discrete?
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    #========================================================================================#
    # Initialize Agent
    #========================================================================================#
    neural_network_args = {
        'actor_n_layers': actor_n_layers,
        'critic_n_layers': critic_n_layers,
        'ob_dim': ob_dim,
        'ac_dim': ac_dim,
        'discrete': discrete,
        'size': size,
        'actor_learning_rate': actor_learning_rate,
        'critic_learning_rate': critic_learning_rate,
        'num_target_updates': num_target_updates,
        'num_grad_steps_per_target_update': num_grad_steps_per_target_update,
        }

    sample_trajectory_args = {
        'animate': animate,
        'max_path_length': max_path_length,
        'min_timesteps_per_batch': min_timesteps_per_batch,
    }

    estimate_advantage_args = {
        'gamma': gamma,
        'normalize_advantages': normalize_advantages,
    }

    agent = Agent(neural_network_args, sample_trajectory_args, estimate_advantage_args)

    #========================================================================================#
    # Training Loop
    #========================================================================================#

    total_timesteps = 0
    for itr in range(n_iter):
        print("********** Iteration %i ************"%itr)
        
        with torch.no_grad(): # use torch.no_grad to disable the gradient calculation
            paths, timesteps_this_batch = agent.sample_trajectories(itr, env)
        total_timesteps += timesteps_this_batch

        # Build arrays for observation, action for the policy gradient update by concatenating 
        # across paths
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_na = np.concatenate([path["action"] for path in paths])
        re_n = np.concatenate([path["reward"] for path in paths])
        next_ob_no = np.concatenate([path["next_observation"] for path in paths])
        terminal_n = np.concatenate([path["terminal"] for path in paths])

        # Call tensorflow operations to:
        # (1) update the critic, by calling agent.update_critic
        # (2) use the updated critic to compute the advantage by, calling agent.estimate_advantage
        # (3) use the estimated advantage values to update the actor, by calling agent.update_actor
        # YOUR CODE HERE
        raise NotImplementedError

        # Log diagnostics
        returns = [path["reward"].sum() for path in paths]
        ep_lengths = [pathlength(path) for path in paths]
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", itr)
        logz.log_tabular("AverageReturn", np.mean(returns))
        logz.log_tabular("StdReturn", np.std(returns))
        logz.log_tabular("MaxReturn", np.max(returns))
        logz.log_tabular("MinReturn", np.min(returns))
        logz.log_tabular("EpLenMean", np.mean(ep_lengths))
        logz.log_tabular("EpLenStd", np.std(ep_lengths))
        logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        logz.dump_tabular()
        logz.save_pytorch_model(agent)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vac')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--actor_learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--critic_learning_rate', '-clr', type=float)
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--num_target_updates', '-ntu', type=int, default=10)
    parser.add_argument('--num_grad_steps_per_target_update', '-ngsptu', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--actor_n_layers', '-l', type=int, default=2)
    parser.add_argument('--critic_n_layers', '-cl', type=int)
    parser.add_argument('--size', '-s', type=int, default=64)
    args = parser.parse_args()

    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = 'ac_' + args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    max_path_length = args.ep_len if args.ep_len > 0 else None

    if not args.critic_learning_rate:
        args.critic_learning_rate = args.actor_learning_rate

    if not args.critic_n_layers:
        args.critic_n_layers = args.actor_n_layers
        
    processes = []

    for e in range(args.n_experiments):
        seed = args.seed + 10*e
        print('Running experiment with seed %d'%seed)

        def train_func():
            train_AC(
                exp_name=args.exp_name,
                env_name=args.env_name,
                n_iter=args.n_iter,
                gamma=args.discount,
                min_timesteps_per_batch=args.batch_size,
                max_path_length=max_path_length,
                actor_learning_rate=args.actor_learning_rate,
                critic_learning_rate=args.critic_learning_rate,
                num_target_updates=args.num_target_updates,
                num_grad_steps_per_target_update=args.num_grad_steps_per_target_update,
                animate=args.render,
                logdir=os.path.join(logdir,'%d'%seed),
                normalize_advantages=not(args.dont_normalize_advantages),
                seed=seed,
                actor_n_layers=args.actor_n_layers,
                critic_n_layers=args.critic_n_layers,
                size=args.size
                )
        p = Process(target=train_func, args=tuple())
        p.start()
        processes.append(p)
        # if you comment in the line below, then the loop will block 
        # until this process finishes
        # p.join()

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
