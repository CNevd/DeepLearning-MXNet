import mxnet as mx

from mxnet.base import _LIB, check_call
from mxnet.base import c_array, mx_uint, mx_float, c_str
from mxnet.ndarray import NDArray, clip

@mx.optimizer.register
class imgNAG(mx.optimizer.Optimizer):
    """SGD with nesterov
    """
    def __init__(self, momentum=0.0, **kwargs):
        super(imgNAG, self).__init__(**kwargs)
        self.momentum = momentum

    def create_state(self, index, weight):
        """Create additional optimizer state such as momentum.

        Parameters
        ----------
        weight : NDArray
            The weight data

        """
        if self.momentum == 0.0:
            return None
        else:
            return zeros(weight.shape, weight.context, dtype=weight.dtype)

    def _get_lr(self, index):
        """get learning rate for index.

        Parameters
        ----------
        index : int
            The index for weight

        Returns
        -------
        lr : float
            learning rate for this index
        """
        mom = 0.0
        if self.lr_scheduler is not None:
            (lr, mom) = self.lr_scheduler(self.num_update)
        else:
            lr = self.lr

        if index in self.lr_mult:
            lr *= self.lr_mult[index]
        elif index in self.idx2name:
            lr *= self.lr_mult.get(self.idx2name[index], 1.0)
        return lr, mom

    def update(self, index, weight, grad, state):
        """Update the parameters.

        Parameters
        ----------
        index : int
            An unique integer key used to index the parameters

        weight : NDArray
            weight ndarray

        grad : NDArray
            grad ndarray

        state : NDArray or other objects returned by init_state
            The auxiliary state used in optimization.
        """
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        (lr, momentum) = self._get_lr(index)
        wd = self._get_wd(index)
        self._update_count(index)

        grad = grad * self.rescale_grad
        if self.clip_gradient is not None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)

        if state:
            mom = state
            mom[:] *= momentum
            grad += wd * weight
            mom[:] += grad
            grad[:] += momentum * mom
            weight[:] += -lr * grad
        else:
            assert self.momentum == 0.0
            weight[:] += -lr * (grad + wd * weight)
