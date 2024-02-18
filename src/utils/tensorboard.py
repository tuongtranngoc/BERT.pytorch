from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


from torch.utils.tensorboard import SummaryWriter

class Tensorboard:
    def __init__(self, config) -> None:
        self.config = config
        outdir = config['tensorboard']
        self.writer = SummaryWriter(outdir)
    
    def add_scalars(self, tag, step, **kwargs):
        for k, v in kwargs.items():
            self.writer.add_scalar(f'{tag}/{k}', v, step)
