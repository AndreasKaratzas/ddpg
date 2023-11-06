
import sys
sys.path.append('../')

import gymnasium as gym
import torch

from pathlib import Path
from gymnasium.wrappers import Monitor

import src.model as model

from src.agent import Agent
from src.args import arguments
from src.logger import HardLogger
from src.utils import parse_configs, update_args, _mlp_configurations, _logger_configuration, info


if __name__ == '__main__':

    # parse arguments
    args = arguments()

    if args.info:
        info()

    if args.config:
        settings = parse_configs(filepath=args.config)
        args = update_args(args, settings)
        args.export_configs = False
    else:
        args.export_configs = True

    ac_kwargs = _mlp_configurations(
        hidden_sizes=args.hidden_sizes, activation=args.activation, extractor=args.arch)
    logger_kwargs = _logger_configuration(
        output_dir=Path(args.checkpoint_dir).parents[0], output_fname=args.logger_name, exp_name=args.name)

    logger = HardLogger(**logger_kwargs, demo=True)
    logger.print_test_message(
        agent="DDPG with" + ("Priority Experience Replay" if args.buffer_arch == 'priority' else "Random Experience Replay") + " and " + args.arch.upper() + " core", 
        env_id=args.env, epochs=args.demo_episodes, device=args.device)
    
    # create RL environment
    env_to_wrap = gym.make(args.env)
    env = Monitor(env_to_wrap, logger.demo_dir, force = True)

    # create the DDPG agent
    agent = Agent(env=env, env_id=args.env, actor_critic=model.MLPActorCritic, ac_kwargs=ac_kwargs, beta=args.beta,
                  seed=args.seed, prior_eps=args.prior_eps, start_steps=args.start_steps, update_every=args.update_every,
                  epochs=args.epochs, replay_size=args.replay_size, gamma=args.gamma, update_after=args.update_after,
                  polyak=args.polyak, auto_save=args.auto_save, elite_criterion=args.elite_criterion, name=args.name, 
                  lr_actor=args.lr_actor, lr_critic=args.lr_critic, batch_size=args.batch_size, alpha=args.alpha,
                  demo_episodes=args.demo_episodes, max_ep_len=args.max_ep_len, logger=logger, act_noise=args.act_noise,
                  checkpoint_freq=args.checkpoint_freq, debug_mode=args.debug_mode, checkpoint_dir=logger.model_dir, 
                  device=args.device, export_configs=args.export_configs, load_checkpoint=args.load_checkpoint, 
                  mu=args.mu, sigma=args.sigma, noise_dist=args.noise_dist, theta=args.theta, buffer_arch=args.buffer_arch)

    agent.load(agent_checkpoint_path=agent.load_checkpoint)

    with torch.no_grad():
        for demo in range(agent.demo_episodes):
            state, done, episode_return, episode_length = agent.env.reset(seed=agent.seed), False, 0, 0
            while not(done or (episode_length == agent.max_ep_len)):
                action = agent.act(state)
                state, reward, done, truncated, e_info = agent.env.step(action)
                episode_return += reward
                episode_length += 1
                agent.env.render()
            # update agent accuracy register
            agent._length.add(episode_length)
            agent._reward.add(episode_return)
    
    env.close()
    env_to_wrap.close()
