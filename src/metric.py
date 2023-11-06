
import sys
sys.path.append('../')

import numpy as np


class MinMaxAvgMeter:
    def __init__(
        self, 
        auto_update: bool = True, 
        update_with_reset: bool = False,
        k_first: int = None,
        k_last: int = None,
        name: str = ''
    ):
        self.auto_update = auto_update
        self.update_with_reset = update_with_reset
        self.k_first = k_first
        self.k_last = k_last
        self.name = name

        self.reset()
    
    def reset(self):
        self.min = np.inf
        self.max = -np.inf
        self.avg = np.nan
        self.reg = []
    
    def add(self, *value):

        if isinstance(*value, (int, float)):
            self.reg.append(*value)
        else:
            val_vec = value[0]
            for val in val_vec:
                self.reg.append(val)

        if self.auto_update:
            self.update()
            
    def min_fn(self):
        if self.k_first is not None:
            if self.k_first < len(self.reg):
                self.min = np.min(self.reg[:self.k_first])
        if self.k_last is not None:
            if self.k_last < len(self.reg):
                self.min = np.min(self.reg[self.k_last:])
        self.min = np.min(self.reg)

    def max_fn(self):
        if self.k_first is not None:
            if self.k_first < len(self.reg):
                self.max = np.max(self.reg[:self.k_first])
        if self.k_last is not None:
            if self.k_last < len(self.reg):
                self.max = np.max(self.reg[self.k_last:])
        self.max = np.max(self.reg)
    
    def avg_fn(self):
        if self.k_first is not None:
            if self.k_first < len(self.reg):
                self.avg = np.mean(self.reg[:self.k_first])
        if self.k_last is not None:
            if self.k_last < len(self.reg):
                self.avg = np.mean(self.reg[self.k_last:])
        self.avg = np.mean(self.reg)
    
    def update(self):
        self.min_fn()
        self.max_fn()
        self.avg_fn()

        if self.update_with_reset:
            self.reset()


class Elitism(MinMaxAvgMeter):
    def __init__(
        self, 
        selection_metric: str = 'avg',
        selection_method: str = 'greater',
        evaluate_with_reset: bool = False,
        k_first: int = None,
        k_last: int = None,
        name: str = ''
    ):
        super(Elitism, self).__init__(
            auto_update=True, update_with_reset=False, k_first=k_first, k_last=k_last, name=name)
        self.selection_metric = selection_metric.lower()
        self.selection_method = selection_method.lower()
        assert self.selection_metric in ['min', 'max', 'avg', 'mean']
        assert self.selection_method in ['lower', 'lower_eq', 'greater', 'greater_eq']

        self.evaluate_with_reset = evaluate_with_reset
        self.placeholder = np.nan
        self.set()
    
    def reset(self, complete=False):
        super().reset()
        if complete:
            self.placeholder = np.nan
    
    def set(self):
        if self.selection_method == 'lower':
            self.is_elite = self.is_lower
        if self.selection_method == 'lower_eq':
            self.is_elite = self.is_lower_eq
        if self.selection_method == 'greater':
            self.is_elite = self.is_greater
        if self.selection_method == 'greater_eq':
            self.is_elite = self.is_greater_eq

    def is_lower(self):
        return self.criterion < self.placeholder
    
    def is_lower_eq(self):
        return self.criterion <= self.placeholder
    
    def is_greater(self):
        return self.criterion > self.placeholder
    
    def is_greater_eq(self):
        return self.criterion >= self.placeholder
    
    def update(self):
        super().update()
        if self.selection_metric in ['avg', 'mean']:
            self.criterion = self.avg
        if self.selection_metric == 'min':
            self.criterion = self.min
        if self.selection_metric == 'max':
            self.criterion = self.max

    def evaluate(self):
        self.update()
        
        if self.placeholder is np.nan or self.is_elite():
            self.placeholder = self.criterion
            return True
    
        if self.evaluate_with_reset:
            self.reset()
        
        return False
