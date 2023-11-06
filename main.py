
import sys
sys.path.append('./')

import gymnasium as gym

import src.model as model

from src.args import arguments
from src.agent import Agent
from src.train import train
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

    # create RL environment
    env = gym.make(args.env)

    ac_kwargs = _mlp_configurations(
        hidden_sizes=args.hidden_sizes, activation=args.activation, extractor=args.arch)
    logger_kwargs = _logger_configuration(
        output_dir=args.checkpoint_dir, output_fname=args.logger_name, exp_name=args.name)
    
    logger = HardLogger(**logger_kwargs)
    logger.print_training_message(
        agent="DDPG with" + ("Priority Experience Replay" if args.buffer_arch == 'priority' else "Random Experience Replay") + " and " + args.arch.upper() + " core", 
        env_id=args.env, epochs=args.epochs, device=args.device, elite_metric=args.elite_criterion, 
        auto_save=(args.elite_criterion.lower() != 'none'))

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

    # train agent
    train(agent=agent)
