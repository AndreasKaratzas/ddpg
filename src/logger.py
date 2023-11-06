
import sys
sys.path.append('../')

import os
import yaml
import logging
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from io import StringIO
from copy import deepcopy
from pathlib import Path


class HardLogger(logging.Logger):
    def __init__(self, output_dir: str = '../data', output_fname: str = None, exp_name: Path = None, demo: bool = False):
        
        self.datetime_tag = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        self.export_data_path = Path(
            output_dir) if output_dir is not None else Path('logs')
        self.name = Path(
            exp_name + '_' + self.datetime_tag) if exp_name is not None else Path(self.datetime_tag)
        self.logger = logging.getLogger(__name__)

        self.parent_dir = self.export_data_path / self.name if not demo else Path(output_dir)
        self.project_path = os.path.abspath(
            os.path.join(__file__, output_dir))

        self.parent_dir_printable_version = str(os.path.abspath(
            self.parent_dir)).replace(':', '').replace('/', ' > ')
        self.project_path_printable_version = str(
            self.project_path).replace(':', '').replace('/', ' > ')

        self.model_dir = self.parent_dir / "model"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.data_dir = self.parent_dir / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = self.parent_dir / "log"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.demo_dir = self.parent_dir / "demo"
        self.demo_dir.mkdir(parents=True, exist_ok=True)

        self.plot_dir = self.parent_dir / "plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        self.log_f_name = self.log_dir / Path(output_fname + ".log") if output_fname else Path("logger.log")
        
        if not demo:
            try:
                f = open(self.log_f_name, "x")
                f.close()
            except:
                raise PermissionError(
                    f"Could not create the file {self.log_f_name}.")

            logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                datefmt='%Y-%m-%dT%H-%M-%S', filename=self.log_f_name, filemode='w')

    def log_message(self, message):
            self.logger.info(message.rstrip())
    
    def export_yaml(self, d=dict(), filename='exp'):
        if filename is None:
            filename = 'exp'
        with open(os.path.join(self.log_dir, filename + '.yaml'), 'w') as yaml_file:
                yaml.dump(d, yaml_file, default_flow_style=False)
    
    def compile_plots(self):
        self.build_log_dataframe()
        self.generate_avg_length_plot()
        self.generate_avg_reward_plot()
        self.generate_avg_q_val_plot()
        self.generate_loss_actor_plot()
        self.generate_loss_critic_plot()
    
    def build_log_dataframe(self):
        df = pd.read_csv(self.log_f_name, delim_whitespace=True, engine='python', skiprows=3, header=0)
        df = df.reset_index()
        prev_cols = list(df.columns)
        df = df.drop(columns=[i for i in prev_cols[:11]])
        prev_cols = list(df.columns)
        df.rename(columns={prev_cols[0]: 'avg_length', prev_cols[1]: 'avg_reward', prev_cols[2]: 'avg_q_val', prev_cols[3]: 'loss_actor', prev_cols[4]: 'loss_critic'}, inplace=True)
        self.df = deepcopy(df)
    
    def generate_avg_length_plot(self):
        fig, axs = plt.subplots(figsize=(12, 12))
        self.df.avg_length.plot(ax=axs)
        fig.patch.set_facecolor('white')
        axs.set_facecolor('xkcd:white')
        axs.set_title('Agent mean achieved test episode length', fontsize=22)
        axs.set_xlabel("Epoch", fontsize=16)
        axs.set_ylabel("Average episode length", fontsize=16)
        fig.tight_layout()
        fig.savefig(self.plot_dir / Path("avg_length.png"))
    
    def generate_avg_reward_plot(self):
        fig, axs = plt.subplots(figsize=(12, 12))
        self.df.avg_reward.plot(ax=axs)
        fig.patch.set_facecolor('white')
        axs.set_facecolor('xkcd:white')
        axs.set_title('Agent mean accumulated test episode reward', fontsize=22)
        axs.set_xlabel("Epoch", fontsize=16)
        axs.set_ylabel("Average accumulated reward", fontsize=16)
        fig.tight_layout()
        fig.savefig(self.plot_dir / Path("avg_reward.png"))
    
    def generate_avg_q_val_plot(self):
        fig, axs = plt.subplots(figsize=(12, 12))
        self.df.avg_q_val.plot(ax=axs)
        fig.patch.set_facecolor('white')
        axs.set_facecolor('xkcd:white')
        axs.set_title('Agent accumulated Q value', fontsize=22)
        axs.set_xlabel("Epoch", fontsize=16)
        axs.set_ylabel("Accumulated Q value", fontsize=16)
        fig.tight_layout()
        fig.savefig(self.plot_dir / Path("avg_q_val.png"))
    
    def generate_loss_actor_plot(self):
        fig, axs = plt.subplots(figsize=(12, 12))
        self.df.loss_actor.plot(ax=axs)
        fig.patch.set_facecolor('white')
        axs.set_facecolor('xkcd:white')
        axs.set_title('Agent actor module loss', fontsize=22)
        axs.set_xlabel("Epoch", fontsize=16)
        axs.set_ylabel("Actor criterion value", fontsize=16)
        fig.tight_layout()
        fig.savefig(self.plot_dir / Path("loss_actor.png"))
    
    def generate_loss_critic_plot(self):
        fig, axs = plt.subplots(figsize=(12, 12))
        self.df.loss_critic.plot(ax=axs)
        fig.patch.set_facecolor('white')
        axs.set_facecolor('xkcd:white')
        axs.set_title('Agent critic module loss', fontsize=22)
        axs.set_xlabel("Epoch", fontsize=16)
        axs.set_ylabel("Critic criterion value", fontsize=16)
        fig.tight_layout()
        fig.savefig(self.plot_dir / Path("loss_critic.png"))

    def print_training_message(
        self,
        agent: str,
        env_id: str,
        epochs: int,
        device: str,
        elite_metric: str,
        auto_save: bool
    ):
        new_line = '\n'
        tab_char = '\t'

        print(f"\n\n\t\tTraining a {colorstr(options=['red', 'underline'], string_args=list([agent]))}\n"
                f"\t\t      in {colorstr(options=['red', 'underline'], string_args=list([env_id]))} environment for {colorstr(options=['red', 'underline'], string_args=list([str(epochs)]))} epochs using\n"
                f"\t\t    a {colorstr(options=['red', 'underline'], string_args=list([device.upper()])) + ' enabled device' if 'cuda' in device.lower() else colorstr(options=['red', 'underline'], string_args=list([device.upper()]))}. {colorstr(options=['blue', 'bold'], string_args=list(['Odysseus']))} will select checkpoints \n"
                f"\t\t        based on {colorstr(options=['red', 'underline'], string_args=list([elite_metric]))}. Auto-saving is {colorstr(options=['blue'], string_args=list(['enabled'])) if auto_save else colorstr(options=['blue'], string_args=list(['disabled']))}{' and' + new_line + tab_char + tab_char + '         the agent will begin learning from step ' + colorstr(options=['red', 'underline'], string_args=list([''])) + '.' + new_line if False else '.' + new_line}"
                f"\n\n\t               The experiment logger is uploaded locally at: \n"
                f"  {colorstr(options=['blue', 'underline'], string_args=list([self.parent_dir_printable_version]))}."
                f"\n\n\t\t                Project absolute path is:\n\t"
                f"{colorstr(options=['blue', 'underline'], string_args=list([self.project_path_printable_version]))}.\n\n")

    def print_test_message(self, agent: str, env_id: str, epochs: int, device: str):
        
        print(f"\n\n\t\t    Evaluating a {colorstr(options=['red', 'underline'], string_args=list([agent]))}\n"
                f"\t\t      in {colorstr(options=['red', 'underline'], string_args=list([env_id]))} environment for {colorstr(options=['red', 'underline'], string_args=list([str(epochs)]))} episodes using \n"
                f"\t\t                 a {colorstr(options=['red', 'underline'], string_args=list([device.upper()])) + ' enabled device' if 'cuda' in device.lower() else colorstr(options=['red', 'underline'], string_args=list([device.upper()]))}."
                f"\n\n\t               The experiment logger is uploaded locally at: \n"
                f"  {colorstr(options=['blue', 'underline'], string_args=list([self.parent_dir_printable_version]))}."
                f"\n\n\t\t                Project absolute path is:\n\t"
                f"{colorstr(options=['blue', 'underline'], string_args=list([self.project_path_printable_version]))}.\n\n")

    def test_logger(self):
        self.logger.debug('debug message')
        self.logger.info('info message')
        self.logger.warning('warn message')
        self.logger.error('error message')
        self.logger.critical('critical message')


class Metrics:
    def __init__(self):
        self.stored_epoch_time = -1
        self.stored_cuda_mem = -1
        self.stored_ram_util = -1
        self.stored_avg_length = -1
        self.stored_avg_reward = -1
        self.stored_avg_q_val = -1
        self.stored_loss_actor = -1
        self.stored_loss_critic = -1
        self.flags = np.zeros(shape=(9))

        self.is_first = True
    
    def set(self, epoch_time, cuda_mem, ram_util, avg_length, avg_reward, avg_q_val, loss_actor, loss_critic):
        self.curr_epoch_time = epoch_time
        self.curr_cuda_mem = cuda_mem
        self.curr_ram_util = ram_util
        self.curr_avg_length = avg_length
        self.curr_avg_reward = avg_reward
        self.curr_avg_q_val = avg_q_val
        self.curr_loss_actor = loss_actor
        self.curr_loss_critic = loss_critic

    def update(self):
        
        if self.is_first:
            self.is_first = False
            self.stored_epoch_time = self.curr_epoch_time
            self.stored_cuda_mem = self.curr_cuda_mem
            self.stored_ram_util = self.curr_ram_util
            self.stored_avg_length = self.curr_avg_length
            self.stored_avg_reward = self.curr_avg_reward
            self.stored_avg_q_val = self.curr_avg_q_val
            self.stored_loss_actor = self.curr_loss_actor
            self.stored_loss_critic = self.curr_loss_critic
        else:
            idx = 1
            if self.curr_epoch_time < self.stored_epoch_time:
                self.flags[idx] = 1
            elif self.curr_epoch_time > self.stored_epoch_time:
                self.flags[idx] = -1
            elif self.curr_epoch_time == self.stored_epoch_time:
                self.flags[idx] = 0

            idx = 2
            if self.curr_cuda_mem < self.stored_cuda_mem:
                self.flags[idx] = 1
            elif self.curr_cuda_mem > self.stored_cuda_mem:
                self.flags[idx] = -1
            elif self.curr_cuda_mem == self.stored_cuda_mem:
                self.flags[idx] = 0
            
            idx = 3
            if self.curr_ram_util < self.stored_ram_util:
                self.flags[idx] = 1
            elif self.curr_ram_util > self.stored_ram_util:
                self.flags[idx] = -1
            elif self.curr_ram_util == self.stored_ram_util:
                self.flags[idx] = 0
            
            idx = 4
            if self.curr_avg_length < self.stored_avg_length:
                self.flags[idx] = 1
            elif self.curr_avg_length > self.stored_avg_length:
                self.flags[idx] = -1
            elif self.curr_avg_length == self.stored_avg_length:
                self.flags[idx] = 0
            
            idx = 5
            if self.curr_avg_reward > self.stored_avg_reward:
                self.flags[idx] = 1
            elif self.curr_avg_reward < self.stored_avg_reward:
                self.flags[idx] = -1
            elif self.curr_avg_reward == self.stored_avg_reward:
                self.flags[idx] = 0
            
            idx = 6
            if self.curr_avg_q_val > self.stored_avg_q_val:
                self.flags[idx] = 1
            elif self.curr_avg_q_val < self.stored_avg_q_val:
                self.flags[idx] = -1
            elif self.curr_avg_q_val == self.stored_avg_q_val:
                self.flags[idx] = 0
            
            idx = 7
            if self.curr_loss_actor < self.stored_loss_actor:
                self.flags[idx] = 1
            elif self.curr_loss_actor > self.stored_loss_actor:
                self.flags[idx] = -1
            elif self.curr_loss_actor == self.stored_loss_actor:
                self.flags[idx] = 0
            
            idx = 8
            if self.curr_loss_critic < self.stored_loss_critic:
                self.flags[idx] = 1
            elif self.curr_loss_critic > self.stored_loss_critic:
                self.flags[idx] = -1
            elif self.curr_loss_critic == self.stored_loss_critic:
                self.flags[idx] = 0
        
            self.stored_epoch_time = self.curr_epoch_time
            self.stored_cuda_mem = self.curr_cuda_mem
            self.stored_ram_util = self.curr_ram_util
            self.stored_avg_length = self.curr_avg_length
            self.stored_avg_reward = self.curr_avg_reward
            self.stored_avg_q_val = self.curr_avg_q_val
            self.stored_loss_actor = self.curr_loss_actor
            self.stored_loss_critic = self.curr_loss_critic

    def compile(self, epoch, epochs, epoch_time, cuda_mem, show_cuda, ram_util, avg_length, avg_reward, avg_q_val, loss_actor, loss_critic):
        self.set(epoch_time, cuda_mem, ram_util, avg_length, avg_reward, avg_q_val, loss_actor, loss_critic)
        self.update()
        self.str = ""
        self.format = ('%9s', '%13s', '%9s',
                       '%9s', '%12s', '%12s', 
                       '%12s', '%12s', '%12s')
        self.msg = tuple()
        self.str += self.format[0]

        for metric in range(1, 9):
            if self.flags[metric] == 1:
                self.str += f"{colorstr(options=['green'], string_args=list([self.format[metric]]))}"
            elif self.flags[metric] == 0:
                self.str += self.format[metric]
            elif self.flags[metric] == -1:
                self.str += f"{colorstr(options=['red'], string_args=list([self.format[metric]]))}"
        
        desc = 'M'
        if len(str(cuda_mem)) - 1 > 6:
            cuda_mem = round(cuda_mem / 1E3, 3)
            desc = 'G'

        for metric in range(9):
            if metric == 0:
                self.msg = self.msg + (f'{epoch + 1}/{epochs}',)
            if metric == 1:
                self.msg = self.msg + (f'{epoch_time:13.3g}',)
            if metric == 2:
                self.msg = self.msg + \
                    (f'{cuda_mem if show_cuda else 0:.3g} ' + desc,)
            if metric == 3:
                self.msg = self.msg + (f'{ram_util}',)
            if metric == 4:
                self.msg = self.msg + (f'{avg_length:12.3g}',)
            if metric == 5:
                self.msg = self.msg + (f'{avg_reward:12.3g}',)
            if metric == 6:
                self.msg = self.msg + (f'{avg_q_val:12.3g}',)
            if metric == 7:
                self.msg = self.msg + (f'{loss_actor:12.3g}',)
            if metric == 8:
                self.msg = self.msg + (f'{loss_critic:12.3g}',)
        
        print((self.str) % self.msg)

        raw_msg = StringIO()
        print((''.join(map(str, self.format))) % self.msg, file=raw_msg)
        raw_msg = raw_msg.getvalue()

        return raw_msg


def colorstr(options, string_args):
    """Usage:
    
    >>> args = ['Andreas', 'Karatzas']
    >>> print(
    ...    f"My name is {colorstr(options=['red', 'underline'], string_args=args)} "
    ...    f"and I like {colorstr(options=['bold', 'cyan'], string_args=list(['Python']))} "
    ...    f"and {colorstr(options=['cyan'], string_args=list(['C++']))}\n")
    Parameters
    ----------
    options : [type]
        [description]
    string_args : [type]
        [description]
    Returns
    -------
    [type]
        [description]
    """
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code
    colors = {'black':          '\033[30m',  # basic colors
              'red':            '\033[31m',
              'green':          '\033[32m',
              'yellow':         '\033[33m',
              'blue':           '\033[34m',
              'magenta':        '\033[35m',
              'cyan':           '\033[36m',
              'white':          '\033[37m',
              'bright_black':   '\033[90m',  # bright colors
              'bright_red':     '\033[91m',
              'bright_green':   '\033[92m',
              'bright_yellow':  '\033[93m',
              'bright_blue':    '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan':    '\033[96m',
              'bright_white':   '\033[97m',
              'end':            '\033[0m',  # miscellaneous
              'bold':           '\033[1m',
              'underline':      '\033[4m'}
    res = []
    for substr in string_args:
        res.append(''.join(colors[x] for x in options) +
                   f'{substr}' + colors['end'])
    space_char = ''.join(colors[x] for x in options) + ' ' + colors['end']
    return space_char.join(res)
