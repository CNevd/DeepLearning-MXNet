"""
learning rate scheduler, which adaptive changes the learning rate based on the
progress
"""
import logging
import mxnet as mx
import math

class Lan_Scheduler(mx.lr_scheduler.LRScheduler):
    """Reduce learning rate in factor

    [Lan, 2012] Guanghui Lan
    An optimal method for stochastic composite optimization

    Assume the weight has been updated by n times, then the learning rate will
    be

    lr = 1/(base_lr + gama * pow(ith_iter, 1/p))

    Parameters
    ----------
    step: int
        schedule learning rate after n updates
    """
    def __init__(self, step, momentum=0.9, stop_factor_lr=1e-8):
        super(Lan_Scheduler, self).__init__()
        if step < 1:
            raise ValueError("Schedule step must be greater or equal than 1 round")
        self.step = step
        self.stop_factor_lr = stop_factor_lr
        self.count = 0
        self.gama = 0.0001
        self.p = 1/1.5
        self.lr = self.base_lr
        self.momentum = self.max_momentum = momentum

    def __call__(self, num_update):
        """
        Call to schedule current learning rate

        Parameters
        ----------
        num_update: int
            the maximal number of updates applied to a weight.
        """

        while num_update > self.count + self.step:
            self.count += self.step
            self.lr = 1 / (self.base_lr + self.gama * math.pow(num_update, self.p))
            if self.lr < self.stop_factor_lr:
                self.lr = self.stop_factor_lr
                logging.info("Update[%d]: now learning rate arrived at %0.5e, will not "
                             "change in the future", num_update, self.lr)
            else:
                logging.info("Update[%d]: Change learning rate to %0.5e",
                             num_update, self.lr)
            xx = -1 - math.log(num_update + 1) / math.log(2)
            self.momentum = min(self.max_momentum, (1 - math.pow(2, xx)))
        return self.lr, self.momentum
