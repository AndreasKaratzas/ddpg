
import sys
sys.path.append('../')

from src.agent import Agent


def test(agent: Agent):

    for demo in range(agent.demo_episodes):
        state, done, episode_return, episode_length = agent.env.reset(seed=agent.seed), False, 0, 0
        while not(done or (episode_length == agent.max_ep_len)):
            action = agent.act(state)
            state, reward, done, truncated, info = agent.env.step(action)
            episode_return += reward
            episode_length += 1
        # update agent accuracy register
        agent._length.add(episode_length)
        agent._reward.add(episode_return)
