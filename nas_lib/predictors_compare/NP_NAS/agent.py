import torch.nn as nn


class BaseAgent(nn.Module):
    def __init__(self, cfg=None):
        super(BaseAgent, self).__init__()
        self.cfg = cfg

    def sample(self, *args, **kargs):
        raise ValueError('subclass should implement this method~~')

    def forward(self, *args, **kargs):
        raise ValueError('subclass should implement this method~~')

    def train_agent(self):
        raise ValueError('subclass should implement this method~~')