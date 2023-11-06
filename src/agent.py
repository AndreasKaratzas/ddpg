
import sys
sys.path.append('../')

import os
import gymnasium as gym
import torch
import numpy as np
import torch.nn.functional as F

from pathlib import Path
from copy import deepcopy
from torch.optim import Adam

from src.logger import HardLogger, colorstr
from src.metric import MinMaxAvgMeter, Elitism
from src.model import MLPActorCritic, count_vars
from src.loss import weighted_mse_loss
from src.priority import PrioritizedReplayBuffer, ReplayBuffer
from src.utils import _seed, zip_strict, compile_experiment_configs
from src.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise, UniformActionNoise


class Agent:
    def __init__(self, env: gym.Env, env_id: str, logger: HardLogger, actor_critic=MLPActorCritic, ac_kwargs=dict(), 
                 replay_size=int(1e6), gamma=0.99, epochs=100, export_configs=False, checkpoint_dir='.', device='cpu', 
                 polyak=0.995, start_steps=10000, lr_actor=1e-3, lr_critic=1e-3, batch_size=128, auto_save=True,
                 update_after=1000, update_every=50, act_noise=0.2, checkpoint_freq=1, name='exp', alpha=0.4,
                 max_ep_len=1000, demo_episodes=10, load_checkpoint='model.pth', beta=0.6, elite_criterion='avg_q_val',
                 debug_mode=False, seed=0, prior_eps=1e-6, mu=0, sigma=0.1, noise_dist='gaussian',
                 theta=0.15, buffer_arch='random'):
        
        self.env = env
        self.env_id = env_id

        # initialize logger
        self.logger = logger

        # hyper-parameters
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.epoch = 0
        self.name = name
        self.beta = beta
        self.timestep = 0
        self.gamma = gamma
        self.alpha = alpha
        self.epochs = epochs
        self.seed = seed
        self.polyak = polyak
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.auto_save = auto_save
        self.act_noise = act_noise
        self.prior_eps = prior_eps
        self.batch_size = batch_size
        self.max_ep_len = max_ep_len
        self.start_steps = start_steps
        self.buffer_arch = buffer_arch
        self.update_after = update_after
        self.update_every = update_every
        self.demo_episodes = demo_episodes
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_freq = checkpoint_freq
        self.elite_criterion = elite_criterion
        self.load_checkpoint = load_checkpoint
        self.noise_dist = noise_dist.lower()
        self.extractor = ac_kwargs['extractor']
        self.activation = ac_kwargs['activation']
        self.hidden_sizes = ac_kwargs['hidden_sizes']

        self.device = torch.device('cuda:0' if torch.cuda.is_available(
        ) else 'cpu') if device is None else torch.device(device=device)

        # seed random number generators for debugging purposes
        if debug_mode:
            _seed(env=self.env, device=self.device, seed=seed)

        # env dims
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape
        
        assert self.noise_dist in ['gaussian', 'uniform', 'ounoise']
        
        self.max_ep_len = max_ep_len
        self.total_steps = self.max_ep_len * (epochs - 1)

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_limit = self.env.action_space.high[0]

        # Create actor-critic module and target networks
        self.online = actor_critic(self.env.observation_space, self.env.action_space, **ac_kwargs)
        self.target = deepcopy(self.online)

        # sync target with online
        self.target.actor.load_state_dict(
            self.online.actor.state_dict())
        self.target.critic.load_state_dict(
            self.online.critic.state_dict())

        # upload models to device
        self.online.actor.to(self.device)
        self.online.critic.to(self.device)
        self.target.actor.to(self.device)
        self.target.critic.to(self.device)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for parameter in self.target.parameters():
            parameter.requires_grad = False
        
        exponent = np.ceil(np.log2(replay_size))
        new_replay_size = 2 ** int(exponent)

        if new_replay_size != replay_size:
            replay_size = new_replay_size
            self.logger.log_message(f"Resetting `replay_size` to {replay_size}" +
                                    f" to be a power of 2 for memory purposes.")
            print(f"\n\t\t           Resetting `{colorstr(options=['red', 'underline'], string_args=['replay_size'])}` to {colorstr(options=['green', 'bold'], string_args=[str(replay_size)])}" +
                  f"\n\t\t          to be a power of 2 for memory purposes.")
        else:
            self.logger.log_message(f"Setting `replay_size` to {replay_size}.")

        assert buffer_arch in ['random', 'priority']
        if self.buffer_arch == 'random':
            self.buffer = ReplayBuffer(
                obs_dim=self.obs_dim, size=replay_size, act_dim=self.act_dim, batch_size=batch_size
            )
        if self.buffer_arch == 'priority':
            # Prioritized Experience Replay
            self.buffer = PrioritizedReplayBuffer(
                obs_dim=self.obs_dim, size=replay_size, act_dim=self.act_dim, batch_size=batch_size, alpha=alpha)

        # Count variables
        var_counts = tuple(count_vars(module) for module in [self.online.actor, self.online.critic])
        self.logger.log_message('\nNumber of parameters: \t actor: %d, \t critic: %d\n' % var_counts)

        # Set up optimizers for policy and q-function
        self.actor_optimizer = Adam(self.online.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = Adam(self.online.critic.parameters(), lr=self.lr_critic)

        self._setup()
        self.replay_size = replay_size
        
        # save the experiment configuration
        if export_configs:
            self.config = compile_experiment_configs(self)
            logger.export_yaml(d=self.config, filename=name)

    def act(self, state):
        """Given a state, choose an action and update value of step.
        Parameters
        ----------
        state : np.ndarray
            A single observation of the current state.
        Returns
        -------
        int
            A float representing the environment action.
        """
        a = self.online.act(torch.as_tensor(
            state, dtype=torch.float32).to(self.device))
        return a

    def cache(self, state, next_state, action, reward, done):
        """Stores the experience replay and priority buffers.
        Parameters
        ----------
        state : numpy.ndarray
            The state of the agent at a time step `t`.
        next_state : numpy.ndarray
            The state of the agent at the next time step `t + 1`.
        action : int
            The action selected by the agent at a time step `t`.
        reward : float
            The reward accumulated by the agent at a time step `t`.
        done : bool
            The terminal indicator at a time step `t`.
        """

        Transition = [state, action, reward, next_state, done]

        # Add a single step transition
        self.buffer.store(*Transition)

    def recall(self):
        """Retrieve a batch of experiences from the experience replay.
        Returns
        -------
        Tuple
            A batch of experiences fetched by the experience replay.
        """

        samples = self.buffer.sample_batch() if self.buffer_arch == 'random' else self.buffer.sample_batch(self.beta)

        state = torch.FloatTensor(samples["obs"]).to(self.device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(self.device)
        action = torch.FloatTensor(samples["acts"]).to(self.device)
        reward = torch.FloatTensor(samples["rews"]).to(self.device)
        done = torch.FloatTensor(samples["done"]).to(self.device)

        weights = None
        indices = None
        if self.buffer_arch == 'priority':
            weights = torch.FloatTensor(samples["weights"]).to(self.device)
            indices = samples["indices"]

        return state, action, reward, next_state, done, weights, indices

    # Set up function for computing DDPG Q-loss
    def compute_td_loss(self, state, action, reward, next_state, done, weights):
        """Computes Cross-entropy loss. This minimizes 
        the Kullback-Leibler divergence (m | p(s_{t}, a_{t})).
        Parameters
        ----------
        state : torch.Tensor
            The state of the agent at a past time step `t`.
        action : torch.Tensor
            The action selected by the agent at a past time step `t`.
        next_state : torch.Tensor
            The state of the agent at the next time step `t + 1`.
        reward : float
            The reward accumulated by the agent at a time step `t`.
        done : bool
            The terminal indicator at a time step `t`.
        weights: np.ndarray
            The importance of each experience.
        Returns
        -------
        Tuple
           * The cross-entropy loss to be backpropagated
           * The Q value return for logging purposes
           * The temporal difference error
        """
        current_q_values = self.online.critic(state, action)

        # Bellman backup for Q function
        with torch.no_grad():
            target_q_values = self.target.critic(next_state, self.target.actor(next_state))
            backup = reward + (1 - done) * self.gamma * target_q_values

        # MSE loss against Bellman backup
        loss_q = ((current_q_values - backup)**2).mean()

        # Compute critic loss
        if self.buffer_arch == 'priority':
            loss_q = weighted_mse_loss(
                input=current_q_values, 
                target=backup, 
                weight=weights
            )
        if self.buffer_arch == 'random':
            loss_q = F.mse_loss(current_q_values, backup)
            
        # Useful info for logging
        td_error = current_q_values - backup
        q_val = current_q_values.detach().cpu().numpy()

        return loss_q, q_val, td_error

    # Set up function for computing DDPG pi loss
    def compute_actor_loss(self, state):
        return -self.online.critic(state, self.online.actor(state)).mean()

    def update(self, state, action, reward, next_state, done, weights):
        self.critic_optimizer.zero_grad()
        # Get q loss
        loss_critic, q_val, td_error = self.compute_td_loss(
            state, action, reward, next_state, done, weights)
        loss_critic.backward()
        self.critic_optimizer.step()

        # Freeze Q-network so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        for parameter in self.online.critic.parameters():
            parameter.requires_grad = False

        # Next run one gradient descent step for pi.
        self.actor_optimizer.zero_grad()
        # Get pi loss
        loss_actor = self.compute_actor_loss(state)
        loss_actor.backward()
        self.actor_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for parameter in self.online.critic.parameters():
            parameter.requires_grad = True
        
        return loss_critic, loss_actor, q_val, td_error
    
    def learn(self):
        if len(self.buffer) < self.batch_size:
            return None
        
        # Sample from memory
        state, action, reward, next_state, done, weights, indices = \
            self.recall()

        loss_critic, loss_actor, q_val, td_error = self.update(
            state, action, reward, next_state, done, weights)

        if self.buffer_arch == 'priority':
            # PER: update priorities
            loss_for_prior = td_error.detach().cpu().numpy()
            new_priorities = np.abs(loss_for_prior) + self.prior_eps
            self.buffer.update_priorities(indices, new_priorities)

            # PER: increase beta
            fraction = min(self.epoch / self.epochs, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

        # update metrics
        self._q_val.add(q_val)
        self._loss_actor.add(loss_actor.item())
        self._loss_critic.add(loss_critic.item())

        self.polyak_update()

    # soft update
    def polyak_update(self):
        with torch.no_grad():
            # zip does not raise an exception if length of parameters does not match.
            for param, target_param in zip_strict(self.online.critic.parameters(), self.target.critic.parameters()):
                target_param.data.mul_(1 - self.polyak)
                torch.add(target_param.data, param.data, alpha=self.polyak, out=target_param.data)
            
            for param, target_param in zip_strict(self.online.actor.parameters(), self.target.actor.parameters()):
                target_param.data.mul_(1 - self.polyak)
                torch.add(target_param.data, param.data, alpha=self.polyak, out=target_param.data)

    def elite_criterion_factory(self) -> bool:
        if not self.auto_save:
            if self.chkpt_cntr % self.checkpoint_freq:
                return True
            self.chkpt_cntr += 1
        
        checkpoint_flag = self._criterion_prt.evaluate()
        
        return checkpoint_flag

    def load(self, agent_checkpoint_path):
        agent_checkpoint_path = Path(agent_checkpoint_path)
        if not agent_checkpoint_path.exists():
            raise ValueError(f"{agent_checkpoint_path} does not exist")

        ckp = torch.load(agent_checkpoint_path, map_location=self.device)

        print(f"Loading model at {agent_checkpoint_path}")

        self.online.actor.load_state_dict(ckp.get('online_actor'))
        self.online.critic.load_state_dict(ckp.get('online_critic'))
        self.target.actor.load_state_dict(ckp.get('target_actor'))
        self.target.critic.load_state_dict(ckp.get('target_critic'))
        self.actor_optimizer.load_state_dict(ckp.get('actor_optimizer'))
        self.critic_optimizer.load_state_dict(ckp.get('critic_optimizer'))

        self._length = ckp.get('_length')
        self._q_val = ckp.get('_q_val')
        self._loss_actor = ckp.get('_loss_actor')
        self._loss_critic = ckp.get('_loss_critic')

        print(
            f"Loaded checkpoint with:"
            f"\n\t * {self._length.avg:7.3f} mean length value"
            f"\n\t * {self._q_val.avg:7.3f} Q value achieved"
            f"\n\t * {self._loss_actor.avg:7.3f} actor model loss"
            f"\n\t * {self._loss_critic.avg:7.3f} critic model loss")

    def store(self):

        if self.elite_criterion_factory():
            torch.save({
                'online_actor': self.online.actor.state_dict(),
                'online_critic': self.online.critic.state_dict(),
                'target_actor': self.target.actor.state_dict(),
                'target_critic': self.target.critic.state_dict(),
                'actor_optimizer': self.actor_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict(),
                '_length': self._length,
                '_q_val': self._q_val,
                '_loss_actor': self._loss_actor,
                '_loss_critic': self._loss_critic
            }, os.path.join(
                self.checkpoint_dir,
                f"epoch_{self.epoch:05d}-" + f"ep_length{self._length.avg:07.3f}-" +
                f"avg_q_val{self._q_val.avg:07.3f}-" + f"avg_reward{self._reward.avg:07.3f}-" +
                f"loss_actor_{self._loss_actor.avg:07.3f}-" +
                f"loss_critic_{self._loss_critic.avg:07.3f}.pth"
            ))

        # hard reset all metrics
        self._length.reset()
        self._reward.reset()
        self._q_val.reset()
        self._loss_actor.reset()
        self._loss_critic.reset()

    # initialize progress metrics
    def _setup(self):
        selection_metric = ''
        self.chkpt_cntr = 0
        if len(self.elite_criterion.split('_')) > 1:
            selection_metric = self.elite_criterion.split('_')[0]
        self._length = MinMaxAvgMeter(name='length')
        self._reward = MinMaxAvgMeter(name='reward')
        self._q_val = MinMaxAvgMeter(name='q_val')
        self._loss_actor = MinMaxAvgMeter(name='loss_actor')
        self._loss_critic = MinMaxAvgMeter(name='loss_critic')

        if self.noise_dist == 'gaussian':
            self.noise_fn = NormalActionNoise(
                mean=np.zeros(self.env.action_space.shape[-1]), 
                sigma=float(self.sigma) * np.ones(self.env.action_space.shape[-1])
            )
        if self.noise_dist == 'ounoise':
            self.noise_fn = OrnsteinUhlenbeckActionNoise(
                mean=np.zeros(self.env.action_space.shape[-1]), 
                sigma=float(self.sigma) * np.ones(self.env.action_space.shape[-1]), 
                theta=self.theta
            )
        if self.noise_dist == 'uniform':
            self.noise_fn = UniformActionNoise(
                low=-self.act_limit,
                high=self.act_limit,
                size=self.act_dim[0]
            )
        
        if 'length' in self.elite_criterion:
            self._length = self._criterion_prt = Elitism(
                selection_metric=selection_metric, selection_method='greater', name='length')
        if 'reward' in self.elite_criterion:
            self._reward = self._criterion_prt = Elitism(
                selection_metric=selection_metric, selection_method='greater', name='reward')
        if 'q_val' in self.elite_criterion:
            self._q_val = self._criterion_prt = Elitism(
                selection_metric=selection_metric, selection_method='greater', name='q_val')
        if 'loss_actor' in self.elite_criterion:
            self._loss_actor = self._criterion_prt = Elitism(
                selection_metric='min', selection_method='lower', name='loss_actor')
        if 'loss_critic' in self.elite_criterion:
            self._loss_critic = self._criterion_prt = Elitism(
                selection_metric='min', selection_method='lower', name='loss_critic')
    