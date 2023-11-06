
import argparse


def arguments():
    """Example run:
        >>> python main.py --env 'MountainCarContinuous-v0' --max-ep-len 1000 --hidden-sizes 256 256 256 --start-steps 1000 --device 'cuda' --debug-mode --name 'MountainCarContinuous-v0' --auto-save --info --logger-name 'MountainCarContinuous-v0' --checkpoint-dir 'data/experiments' --batch-size 256
    """
    parser = argparse.ArgumentParser()
    # based on original implementation: https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/her/experiment/config.py#L17
    parser.add_argument('--env', type=str,
                        help='Environment name. The environment must satisfy the OpenAI Gym API.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to run and train agent (default: 200).')
    parser.add_argument('--start-steps', type=int, default=10000,
                        help=' Number of steps for uniform-random action selection, before running real policy. Helps exploration. (default: 10000).')
    parser.add_argument('--update-every', type=int, default=50,
                        help='Number of env interactions that should elapse between gradient descent updates. (default: 50).')
    parser.add_argument('--update-after', type=int, default=1000,
                        help='Number of env interactions to collect before starting to do gradient descent updates. Ensures replay buffer is full enough for useful updates. (default: 1000).')
    parser.add_argument('--checkpoint-freq', type=int, default=5, 
                        help='Frequency for saving the framework state in epochs (default: 5).')
    parser.add_argument('--seed', type=int, default=0, 
                        help='Seed for any Random Number Generators (default: 0).')
    parser.add_argument('--checkpoint-dir', type=str, default='data', 
                        help='Used in configuring the logger, to decide where to store experiment results (default: `data`).')
    parser.add_argument('--replay-size', type=int, default=int(1e6), 
                        help='Maximum length of replay buffer (default: 1e6).')
    parser.add_argument('--batch-size', type=int, default=128, 
                        help='Mini-batch size for training (default: 128).')
    parser.add_argument('--gamma', type=float, default=0.99, 
                        help='Discount factor used for Q learning updates (default: 0.99).')
    parser.add_argument('--prior-eps', type=float, default=1e-6, 
                        help='Priority epsilon. Guarantees every transition can be sampled (default: 1e-6).')
    parser.add_argument('--act-noise', type=float, default=0.2, 
                        help='Target policy noise added (default: 0.2).')
    parser.add_argument('--alpha', type=float, default=0.2,
                        help='Determines how much prioritization is used (default: 0.2).')
    parser.add_argument('--beta', type=float, default=0.6, 
                        help='Determines how much importance sampling is used (default: 0.6).')
    parser.add_argument('--lr-actor', type=float, default=1e-3, 
                        help='Learning rate for policy (default: 1e-3).')
    parser.add_argument('--lr-critic', type=float, default=1e-3, 
                        help='Learning rate for Q-networks (default: 1e-3).')
    parser.add_argument('--polyak', type=float, default=0.95,
                        help=f"Interpolation factor in polyak averaging for target networks.")
    parser.add_argument('--demo-episodes', type=int, default=5, 
                        help='Number of episodes for demonstration (default: 5).')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Choose a device to utilize for the experiment (default: CPU).')
    parser.add_argument('--debug-mode', action='store_true',
                        help='Pass this option to seed every random number generator.')
    parser.add_argument('--name', type=str, help='A name to associate with the experiment.')
    parser.add_argument('--info', action='store_true',
                        help='Pass this option to print information about the project.')
    parser.add_argument('--auto-save', action='store_true',
                        help=f'Pass this option to compile a checkpoint only if the agent improves. ' +
                             f'Default action: checkpoint the agent in every epoch.')
    parser.add_argument('--elite-criterion', type=str, default='avg_reward',
                        help=f'The metric that indicates agent improvement.' +
                             f'Options:\n\t1. `none`\n\t2. `avg_length`\n\t ' +
                             f'3. `avg_q_val`' +
                             f'\n\t4. `loss_actor`\n\t5. `loss_critic`\n\t' + 
                             f'\n\t6. `avg_reward`')
    parser.add_argument('--load-checkpoint', type=str, 
                        help='Load a pretrained model from that filepath (example: `data/model.pth`).')
    parser.add_argument('--logger-name', type=str, help='A logger filename.')
    parser.add_argument('--arch', type=str, default='mlp', help='The preferred model architecture for the compiled models.')
    parser.add_argument('--hidden-sizes', type=int, nargs='+', default=(256, 256),
                        help='Number of neurons for each hidden layer.')
    parser.add_argument('--activation', default='relu', help='The activation function for the hidden layers of the model.')
    parser.add_argument('--max-ep-len', default=1000, type=int, help='An upper limit indicating the maximum number of agent time steps per cycle.')
    # Example: `--config './data/CartPole-V0.yaml'`
    parser.add_argument('--config', type=str, help=f'(Optional) A configurations file. ' +
                                                   f'Use the configurations file to overwrite some of ' +
                                                   f'CL arguments.')
    parser.add_argument('--mu', type=float, default=0,
                        help=f"Mean for the Gaussian distribution sampled to yield the noise factor (default: 0).")
    parser.add_argument('--sigma', type=float, default=0.1,
                        help=f"Standard deviation for the Gaussian distribution sampled to yield the noise factor (default: 0.1).")
    parser.add_argument('--theta', type=float, default=0.15,
                        help=f"Theta for Ornstein-Uhlenbeck process (default: 0.15).")
    parser.add_argument('--noise-dist', type=str, default='uniform',
                        help='Define the format of the noise distribution (default: `uniform`). Supported:\n\t1. `gaussian`\n\t2. `uniform`\n\t `ounoise`')
    parser.add_argument('--buffer-arch', type=str, default='random',
                        help='Define the architecture of the experience replay buffer (default: `random`). Supported:\n\t1. `random`\n\t2. `priority`')
    args = parser.parse_args()

    return args
