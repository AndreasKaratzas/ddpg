

import sys
sys.path.append('../')

import numpy as np


class ReplayBuffer:
    """Random experience replay buffer.
    Attributes
    ----------
    obs_dim : Tuple
        Dimensions of a random observation vector.
    act_dim : Tuple
        Dimensions of a random action vector.
    size : int
        Number of experiences to hold.
    batch_size: int
        Number of samples to batchify.
    """

    def __init__(self, obs_dim, size, act_dim, batch_size):
        self.obs_buf = np.zeros([size, *obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, *obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, *act_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros([size], dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def store(self, obs, act, rew, next_obs, done):
        """Stores experience.
        
        Attributes
        ----------
        obs : np.ndarray
            State of agent at a time step `t`.
        act : int
            Selected action by the agent at a time step `t`.
        rew : float
            Accumulated reward by the agent after an action 
            given its state at a time step `t`.
        next_obs : np.ndarray
            State of the agent at a time step `t + 1`.
        done : bool
            Terminal flag after the selected action at a time step `t`.
        """

        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self):
        """Sample a batch of experiences.
        
        Returns
        -------
        Dict
            A dictionary with a number of past experiences equal to the batch number
            fetched by the random replay algorithm.
        """

        idxs = np.random.choice(
            self.size, size=self.batch_size, replace=False)

        return dict(obs=self.obs_buf[idxs], next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs], rews=self.rews_buf[idxs], done=self.done_buf[idxs])

    def __len__(self):
        return self.size
