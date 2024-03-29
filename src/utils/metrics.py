from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


class BatchMeter(object):
    """Calculate average/sum value after each time
    """
    def __init__(self) -> None:
        self.reset()
    
    def reset(self):
        self.avg = 0
        self.sum = 0
        self.value = 0
        self.count = 0
    
    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count
        
    def get_value(self, summary_type='mean'):
        if summary_type == 'mean':
            return self.avg
        elif summary_type == 'sum':
            return self.sum
        else:
           return self.value