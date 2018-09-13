"""
Original code from John Schulman for CS294 Deep Reinforcement Learning Spring 2017
Adapted for CS294-112 Fall 2017 by Abhishek Gupta and Joshua Achiam
Adapted for CS294-112 Fall 2018 by Michael Chang and Soroush Nasiriany
Modified to pytorch version by Ning Dai
"""
import numpy as np
import torch
import gym
import logz
import scipy.signal
import os
import time
import inspect
from torch.multiprocessing import Process
from torch import nn, optim

#============================================================================================#
# Utilities
#============================================================================================#

#========================================================================================#
#                           ----------PROBLEM 2----------
#========================================================================================#  
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
    # YOUR CODE HERE
    raise NotImplementedError
    return nn.Sequential(*layers)

def pathlength(path):
    return len(path["reward"])

def setup_logger(logdir, locals_):
    # Configure output directory for logging
    logz.configure_output_dir(logdir)
    # Log experimental parameters
    args = inspect.getargspec(train_PG)[0]
    hyperparams = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_hyperparams(hyperparams)

class PolicyNet(nn.Module):
    def __init__(self, neural_network_args):
        super(PolicyNet, self).__init__()
        self.ob_dim = neural_network_args['ob_dim']
        self.ac_dim = neural_network_args['ac_dim']
        self.discrete = neural_network_args['discrete']
        self.hidden_size = neural_network_args['size']
        self.n_layers = neural_network_args['n_layers']

        self.define_model_components()
        
    #========================================================================================#
    #                           ----------PROBLEM 2----------
    #========================================================================================#
    def define_model_components(self):
        """
            Define the parameters of policy network here.
            You can use any instance of nn.Module or nn.Parameter.

            Hint: use the 'build_mlp' function defined above
                In the discrete case, model should output logits of a categorical distribution
                    over the actions
                In the continuous case, model should output a tuple (mean, log_std) of a Gaussian
                    distribution over actions. log_std should just be a trainable
                    variable, not a network output.
        """
        # YOUR_CODE_HERE
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
            # YOUR_CODE_HERE
            ts_logits_na = None
            return ts_logits_na
        else:
            # YOUR_CODE_HERE
            ts_mean = None
            ts_logstd = None
            return (ts_mean, ts_logstd)
    
#============================================================================================#
# Policy Gradient
#============================================================================================#

class Agent(object):
    def __init__(self, neural_network_args, sample_trajectory_args, estimate_return_args):
        super(Agent, self).__init__()
        self.ob_dim = neural_network_args['ob_dim']
        self.ac_dim = neural_network_args['ac_dim']
        self.discrete = neural_network_args['discrete']
        self.hidden_size = neural_network_args['size']
        self.n_layers = neural_network_args['n_layers']
        self.learning_rate = neural_network_args['learning_rate']

        self.animate = sample_trajectory_args['animate']
        self.max_path_length = sample_trajectory_args['max_path_length']
        self.min_timesteps_per_batch = sample_trajectory_args['min_timesteps_per_batch']

        self.gamma = estimate_return_args['gamma']
        self.reward_to_go = estimate_return_args['reward_to_go']
        self.nn_baseline = estimate_return_args['nn_baseline']
        self.normalize_advantages = estimate_return_args['normalize_advantages']

        self.policy_net = PolicyNet(neural_network_args)
        params = list(self.policy_net.parameters())

        #========================================================================================#
        #                           ----------PROBLEM 6----------
        # Optional Baseline
        #
        # Define a neural network baseline.
        #========================================================================================#
        if self.nn_baseline:
            self.value_net = build_mlp(self.ob_dim, 1, self.n_layers, self.hidden_size)
            params += list(self.value_net.parameters())

        self.optimizer = optim.Adam(params, lr=self.learning_rate)
        
    #========================================================================================#
    #                           ----------PROBLEM 2----------
    #========================================================================================#
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
            # YOUR_CODE_HERE
            ts_sampled_ac = None
        else:
            ts_mean, ts_logstd = self.policy_net(ts_ob_no)
            # YOUR_CODE_HERE
            ts_sampled_ac = None

        sampled_ac = ts_sampled_ac.numpy()
        return sampled_ac

    #========================================================================================#
    #                           ----------PROBLEM 2----------
    #========================================================================================#
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
            # YOUR_CODE_HERE
            ts_logprob_n = None
        else:
            ts_mean, ts_logstd = policy_parameters
            # YOUR_CODE_HERE
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
        obs, acs, rewards = [], [], []
        steps = 0
        while True:
            if animate_this_episode:
                env.render()
                time.sleep(0.1)
            obs.append(ob)
            #====================================================================================#
            #                           ----------PROBLEM 3----------
            #====================================================================================#
            raise NotImplementedError
            ac = None # YOUR CODE HERE
            ac = ac[0]
            acs.append(ac)
            ob, rew, done, _ = env.step(ac)
            rewards.append(rew)
            steps += 1
            if done or steps > self.max_path_length:
                break
        path = {"observation" : np.array(obs, dtype=np.float32), 
                "reward" : np.array(rewards, dtype=np.float32), 
                "action" : np.array(acs, dtype=np.float32)}
        return path

    #====================================================================================#
    #                           ----------PROBLEM 3----------
    #====================================================================================#
    def sum_of_rewards(self, re_n):
        """
            Monte Carlo estimation of the Q function.

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from 
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories

            arguments:
                re_n: length: num_paths. Each element in re_n is a numpy array 
                    containing the rewards for the particular path

            returns:
                q_n: shape: (sum_of_path_lengths). A single vector for the estimated q values 
                    whose length is the sum of the lengths of the paths

            ----------------------------------------------------------------------------------
            
            Your code should construct numpy arrays for Q-values which will be used to compute
            advantages (which will in turn be fed to the placeholder you defined in 
            Agent.define_placeholders). 
            
            Recall that the expression for the policy gradient PG is
            
                  PG = E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * (Q_t - b_t )]
            
            where 
            
                  tau=(s_0, a_0, ...) is a trajectory,
                  Q_t is the Q-value at time t, Q^{pi}(s_t, a_t),
                  and b_t is a baseline which may depend on s_t. 
            
            You will write code for two cases, controlled by the flag 'reward_to_go':
            
              Case 1: trajectory-based PG 
            
                  (reward_to_go = False)
            
                  Instead of Q^{pi}(s_t, a_t), we use the total discounted reward summed over 
                  entire trajectory (regardless of which time step the Q-value should be for). 
            
                  For this case, the policy gradient estimator is
            
                      E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * Ret(tau)]
            
                  where
            
                      Ret(tau) = sum_{t'=0}^T gamma^t' r_{t'}.
            
                  Thus, you should compute
            
                      Q_t = Ret(tau)
            
              Case 2: reward-to-go PG 
            
                  (reward_to_go = True)
            
                  Here, you estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting
                  from time step t. Thus, you should compute
            
                      Q_t = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
            
            
            Store the Q-values for all timesteps and all trajectories in a variable 'q_n',
            like the 'ob_no' and 'ac_na' above. 
        """
        # YOUR_CODE_HERE
        if self.reward_to_go:
            raise NotImplementedError
        else:
            raise NotImplementedError
        return q_n

    def compute_advantage(self, ob_no, q_n):
        """
            Computes advantages by (possibly) subtracting a baseline from the estimated Q values

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from 
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                q_n: shape: (sum_of_path_lengths). A single vector for the estimated q values 
                    whose length is the sum of the lengths of the paths

            returns:
                adv_n: shape: (sum_of_path_lengths). A single vector for the estimated 
                    advantages whose length is the sum of the lengths of the paths
        """
        #====================================================================================#
        #                           ----------PROBLEM 6----------
        # Computing Baselines
        #====================================================================================#
        if self.nn_baseline:
            # If nn_baseline is True, use your neural network to predict reward-to-go
            # at each timestep for each trajectory, and save the result in a variable 'b_n'
            # like 'ob_no', 'ac_na', and 'q_n'.
            #
            # Hint #bl1: rescale the output from the nn_baseline to match the statistics
            # (mean and std) of the current batch of Q-values. (Goes with Hint
            # #bl2 in Agent.update_parameters.
            raise NotImplementedError
            # YOUR CODE HERE
            b_n = None 
            adv_n = q_n - b_n
        else:
            adv_n = q_n.copy()
        return adv_n

    def estimate_return(self, ob_no, re_n):
        """
            Estimates the returns over a set of trajectories.

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from 
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                re_n: length: num_paths. Each element in re_n is a numpy array 
                    containing the rewards for the particular path

            returns:
                q_n: shape: (sum_of_path_lengths). A single vector for the estimated q values 
                    whose length is the sum of the lengths of the paths
                adv_n: shape: (sum_of_path_lengths). A single vector for the estimated 
                    advantages whose length is the sum of the lengths of the paths
        """
        q_n = self.sum_of_rewards(re_n)
        adv_n = self.compute_advantage(ob_no, q_n)
        #====================================================================================#
        #                           ----------PROBLEM 3----------
        # Advantage Normalization
        #====================================================================================#
        if self.normalize_advantages:
            # On the next line, implement a trick which is known empirically to reduce variance
            # in policy gradient methods: normalize adv_n to have mean zero and std=1.
            raise NotImplementedError
            adv_n = None # YOUR_CODE_HERE
        return q_n, adv_n

    def update_parameters(self, ob_no, ac_na, q_n, adv_n):
        """ 
            Update the parameters of the policy and (possibly) the neural network baseline, 
            which is trained to approximate the value function.

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                ac_na: shape: (sum_of_path_lengths).
                q_n: shape: (sum_of_path_lengths). A single vector for the estimated q values 
                    whose length is the sum of the lengths of the paths
                adv_n: shape: (sum_of_path_lengths). A single vector for the estimated 
                    advantages whose length is the sum of the lengths of the paths

            returns:
                nothing

        """
        # convert numpy array to pytorch tensor
        ts_ob_no, ts_ac_na, ts_q_n, ts_adv_n = map(lambda x: torch.from_numpy(x), [ob_no, ac_na, q_n, adv_n])

        # The policy takes in an observation and produces a distribution over the action space
        policy_parameters = self.policy_net(ts_ob_no)

        # We can compute the logprob of the actions that were actually taken by the policy
        # This is used in the loss function.
        ts_logprob_n = self.get_log_prob(policy_parameters, ts_ac_na)

        # clean the gradient for model parameters
        self.optimizer.zero_grad()
        
        #========================================================================================#
        #                           ----------PROBLEM 3----------
        # Loss Function for Policy Gradient
        #========================================================================================#
        raise NotImplementedError
        loss = None # YOUR CODE HERE
        loss.backward()
        
        #====================================================================================#
        #                           ----------PROBLEM 6----------
        # Optimizing Neural Network Baseline
        #====================================================================================#
        if self.nn_baseline:
            # If a neural network baseline is used, set up the targets and the output of the 
            # baseline. 
            # 
            # Fit it to the current batch in order to use for the next iteration. Use the 
            # self.value_net you defined earlier.
            #
            # Hint #bl2: Instead of trying to target raw Q-values directly, rescale the 
            # targets to have mean zero and std=1. (Goes with Hint #bl1 in 
            # Agent.compute_advantage.)

            # YOUR_CODE_HERE
            raise NotImplementedError
            baseline_prediction = None
            ts_target_n = None
            baseline_loss = None
            baseline_loss.backward()

        #====================================================================================#
        #                           ----------PROBLEM 3----------
        # Performing the Policy Update
        #====================================================================================#

        # Call the optimizer to perform the policy gradient update based on the current batch 
        # of rollouts.
        # 
        # For debug purposes, you may wish to save the value of the loss function before
        # and after an update, and then log them below. 

        # YOUR_CODE_HERE
        raise NotImplementedError

def train_PG(
        exp_name,
        env_name,
        n_iter, 
        gamma, 
        min_timesteps_per_batch, 
        max_path_length,
        learning_rate, 
        reward_to_go, 
        animate, 
        logdir, 
        normalize_advantages,
        nn_baseline, 
        seed,
        n_layers,
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
        'n_layers': n_layers,
        'ob_dim': ob_dim,
        'ac_dim': ac_dim,
        'discrete': discrete,
        'size': size,
        'learning_rate': learning_rate,
        }

    sample_trajectory_args = {
        'animate': animate,
        'max_path_length': max_path_length,
        'min_timesteps_per_batch': min_timesteps_per_batch,
    }

    estimate_return_args = {
        'gamma': gamma,
        'reward_to_go': reward_to_go,
        'nn_baseline': nn_baseline,
        'normalize_advantages': normalize_advantages,
    }

    agent = Agent(neural_network_args, sample_trajectory_args, estimate_return_args)

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
        re_n = [path["reward"] for path in paths]

        with torch.no_grad():
            q_n, adv_n = agent.estimate_return(ob_no, re_n)
            
        agent.update_parameters(ob_no, ac_na, q_n, adv_n)

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
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=64)
    args = parser.parse_args()

    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    max_path_length = args.ep_len if args.ep_len > 0 else None

    processes = []

    for e in range(args.n_experiments):
        seed = args.seed + 10*e
        print('Running experiment with seed %d'%seed)

        def train_func():
            train_PG(
                exp_name=args.exp_name,
                env_name=args.env_name,
                n_iter=args.n_iter,
                gamma=args.discount,
                min_timesteps_per_batch=args.batch_size,
                max_path_length=max_path_length,
                learning_rate=args.learning_rate,
                reward_to_go=args.reward_to_go,
                animate=args.render,
                logdir=os.path.join(logdir,'%d'%seed),
                normalize_advantages=not(args.dont_normalize_advantages),
                nn_baseline=args.nn_baseline, 
                seed=seed,
                n_layers=args.n_layers,
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
