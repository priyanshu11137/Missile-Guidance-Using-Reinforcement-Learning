# %%
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense
import tensorflow.keras as K
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as Kera
import matplotlib.pyplot as plt
import math
import random
from tensorflow.keras import regularizers
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
import os
import time
from collections import deque
from scipy import signal
# %%
# tf.compat.v1.disable_eager_execution()
class MemoryPPO:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    This is adapted version of the Spinning Up Open Ai PPO code of the buffer.
    https://github.com/openai/spinningup/blob/master/spinup/algos/ppo/ppo.py
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        # a fucntion so that different dimensions state array shapes are all processed corecctly
        def combined_shape(length, shape=None):
            if shape is None:
                return (length,)
            return (length, shape) if np.isscalar(shape) else (length, *shape)
        # just empty arrays with appropriate sizes
        self.obs_buf = np.zeros(combined_shape(
            size, obs_dim), dtype=np.float32)  # states
        self.act_buf = np.zeros(combined_shape(
            size, act_dim), dtype=np.float32)  # actions
        # actual rwards from state using action
        self.rew_buf = np.zeros(size, dtype=np.float32)
        # predicted values of state
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)  # gae advantewages
        self.ret_buf = np.zeros(size, dtype=np.float32)  # discounted rewards
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def discount_cumsum(self, x, discount):
        """
        magic from rllab for computing discounted cumulative sums of vectors.
        example input: [x0, x1, x2] output: [x0 + discount * x1 + discount^2 * x2,  x1 + discount * x2, x2]
        """
        return signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def store(self, obs, act, rew, val):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.ptr += 1

    def finish_path(self, last_val=0):
        """Finishes an episode of data collection by calculating the diffrent rewards and resetting pointers.
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self.discount_cumsum(
            deltas, self.gamma * self.lam)
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = self.discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get_batch(self, batch_size):
        """simply retuns a randomized batch of batch_size from the data in memory
        """
        # make a randlim list with batch_size numbers.
        pos_lst = np.random.randint(self.ptr, size=batch_size)
        return self.obs_buf[pos_lst], self.act_buf[pos_lst], self.adv_buf[pos_lst], self.ret_buf[pos_lst], self.val_buf[pos_lst]

    def clear(self):
        """Set back pointers to the beginning
        """
        self.ptr, self.path_start_idx = 0, 0
