
import numpy as np
import sys
sys.path.append('../')

import time
import torch
import psutil
import torchnet.meter as tnt

from src.agent import Agent
from src.test import test
from src.scaler import scale_action, unscale_action
from src.logger import Metrics


def train(agent: Agent):

     # initialize logger placeholders
    epoch_time = tnt.AverageValueMeter()
    metrics = Metrics()
    ep_start = time.time()
    init_msg = f"{'Epoch':>9}{'epoch_time':>13}{'gpu_mem':>9}{'ram_util':>9}{'avg_length':>12}{'avg_reward':>12}{'avg_q_val':>12}{'loss_actor':>12}{'loss_critic':>12}"
    print("\n\n" + init_msg)
    agent.logger.log_message(init_msg)

    state, episode_return, episode_length = agent.env.reset(), 0, 0
    # Main loop: collect experience in env and update/log each epoch
    for timestep in range(agent.total_steps):
        
        # update agent timestep
        agent.timestep = timestep

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy (with some noise).
        if timestep > agent.start_steps:
            unscaled_action = agent.act(state)
        else:
            unscaled_action = agent.env.action_space.sample()
        
        scaled_action = scale_action(action_space=agent.env.action_space, action=unscaled_action)
        scaled_action = np.clip(scaled_action + agent.noise_fn(), -1, 1)
        # We store the scaled action in the buffer
        buffer_action = scaled_action
        action = unscale_action(action_space=agent.env.action_space, action=scaled_action)

        # Step the env
        next_state, reward, done, _ = agent.env.step(action)
        episode_return += reward
        episode_length += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        done = False if episode_length == agent.max_ep_len else done

        # Store experience to replay buffer
        agent.cache(state, next_state, buffer_action, reward, done)
        
        # update most recent observation
        state = next_state

        # End of trajectory handling
        if done or (episode_length == agent.max_ep_len):
            state, episode_return, episode_length = agent.env.reset(), 0, 0
            agent.noise_fn.reset()

        # Update handling
        if timestep >= agent.update_after and timestep % agent.update_every == 0:
            for _ in range(agent.update_every):
                agent.learn()

        # End of epoch handling
        if (timestep+1) % agent.max_ep_len == 0:
            epoch = (timestep+1) // agent.max_ep_len

            # Test the performance of the deterministic version of the agent.
            test(agent)

            if epoch != agent.epoch and timestep >= agent.update_after:
                agent.epoch = epoch
                ep_end = time.time()
                epoch_time.add(ep_end - ep_start)

                # log agent progress
                msg = metrics.compile(
                    epoch=epoch,
                    epochs=agent.epochs,
                    epoch_time=round(epoch_time.mean, 3),
                    cuda_mem=round(torch.cuda.memory_reserved() / 1E6, 3) if agent.device.type == 'cuda' else 0, 
                    show_cuda=agent.device.type == 'cuda',
                    ram_util=psutil.virtual_memory().percent,
                    avg_length=round(agent._length.avg, 3), 
                    avg_reward=round(agent._reward.avg, 3),
                    avg_q_val=round(agent._q_val.avg, 3), 
                    loss_actor=round(np.abs(agent._loss_actor.avg), 3), 
                    loss_critic=round(agent._loss_critic.avg, 3)
                )

                agent.store()
                agent.logger.log_message(msg)
                ep_start = time.time()

    agent.logger.compile_plots()
