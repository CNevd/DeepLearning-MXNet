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
    def __init__(self, step, stop_factor_lr=1e-8):
        super(Lan_Scheduler, self).__init__()
        if step < 1:
            raise ValueError("Schedule step must be greater or equal than 1 round")
        self.step = step
        self.stop_factor_lr = stop_factor_lr
        self.count = 0
        self.gama = 0.0001
        self.p = 1/1.5

    def __call__(self, num_update):
        """
        Call to schedule current learning rate

        Parameters
        ----------
        num_update: int
            the maximal number of updates applied to a weight.
        """

        if num_update > self.count + self.step:
            self.count += self.step
            lr = 1 / (self.base_lr + self.gama * math.pow(num_update/self.step, self.p))
            if lr < self.stop_factor_lr:
                lr = self.stop_factor_lr
                logging.info("Update[%d]: now learning rate arrived at %0.5e, will not "
                             "change in the future", num_update, lr)
            else:
                logging.info("Update[%d]: Change learning rate to %0.5e",
                             num_update, lr)
        else:
            lr = self.base_lr
        return lr
