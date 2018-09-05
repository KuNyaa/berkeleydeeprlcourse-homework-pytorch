import pickle
import numpy as np
from functools import reduce


def load_policy(filename):
    def read_layer(l):
        assert list(l.keys()) == ['AffineLayer']
        assert sorted(l['AffineLayer'].keys()) == ['W', 'b']
        W, b = l['AffineLayer']['W'].astype(np.float32), l['AffineLayer']['b'].astype(np.float32)
        return lambda x: np.matmul(x, W) + b
        
    def build_nonlin_fn(nonlin_type):
        if nonlin_type == 'lrelu':
            leak = 0.01 # openai/imitation nn.py:233
            return lambda x: 0.5 * (1 + leak) * x + 0.5 * (1 - leak) * np.abs(x)
        elif nonlin_type == 'tanh':
            return lambda x: np.tanh(x)
        else:
            raise NotImplementedError(nonlin_type)
    
    with open(filename, 'rb') as f:
        data = pickle.loads(f.read())

    # assert len(data.keys()) == 2
    nonlin_type = data['nonlin_type']
    nonlin_fn = build_nonlin_fn(nonlin_type)
    policy_type = [k for k in data.keys() if k != 'nonlin_type'][0]

    assert policy_type == 'GaussianPolicy', 'Policy type {} not supported'.format(policy_type)
    policy_params = data[policy_type]

    assert set(policy_params.keys()) == {'logstdevs_1_Da', 'hidden', 'obsnorm', 'out'}
    
    # Build observation normalization layer
    assert list(policy_params['obsnorm'].keys()) == ['Standardizer']
    obsnorm_mean = policy_params['obsnorm']['Standardizer']['mean_1_D']
    obsnorm_meansq = policy_params['obsnorm']['Standardizer']['meansq_1_D']
    obsnorm_stdev = np.sqrt(np.maximum(0, obsnorm_meansq - np.square(obsnorm_mean)))
    #print('obs', obsnorm_mean.shape, obsnorm_stdev.shape)

    
    # Build hidden layers
    assert list(policy_params['hidden'].keys()) == ['FeedforwardNet']
    layer_params = policy_params['hidden']['FeedforwardNet']
    layers = []
    for layer_name in sorted(layer_params.keys()):
        l = layer_params[layer_name]
        fc_layer = read_layer(l)
        layers += [fc_layer, nonlin_fn]

    # Build output layer
    fc_layer = read_layer(policy_params['out'])
    layers += [fc_layer]
    layers_forward = lambda inp: reduce(lambda x, fn: fn(x), [inp] + layers)
    
    
    def forward_pass(obs):
        ''' Build the forward pass for policy net.

        Input: batched observation. (shape: [batch_size, obs_dim])

        Output: batched action. (shape: [batch_size, action_dim])
        '''
        obs = obs.astype(np.float32)
        normed_obs = (obs - obsnorm_mean) / (obsnorm_stdev + 1e-6) # 1e-6 constant from Standardizer class in nn.py:409 in openai/imitation
        output = layers_forward(normed_obs.astype(np.float32))

        return output

    return forward_pass
