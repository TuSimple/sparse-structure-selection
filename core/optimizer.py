import sys
import numpy as np
import mxnet as mx
from math import sqrt
from mxnet.optimizer import Optimizer, SGD, clip
from mxnet.ndarray import NDArray, zeros
from mxnet.ndarray import sgd_update, sgd_mom_update


@mx.optimizer.register
class APGNAG(SGD):
    """APG and NAG.
    """
    def __init__(self, lambda_name=None, gamma=None, **kwargs):
        super(APGNAG, self).__init__(**kwargs)
        self.lambda_name = lambda_name
        self.gamma = gamma

    def update(self, index, weight, grad, state):
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        self._update_count(index)

        grad = grad * self.rescale_grad
        if self.clip_gradient is not None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)

        if self.idx2name[index].startswith(self.lambda_name):
            # APG
            if state is not None:
                mom = state
                mom[:] *= self.momentum
                z = weight - lr * grad # equ 10
                z = self.soft_thresholding(z, lr * self.gamma)
                mom[:] = z - weight + mom # equ 11
                weight[:] = z + self.momentum * mom # equ 12
            else:
                assert self.momentum == 0.0
            # no-negative
            weight[:] = mx.ndarray.maximum(0.0, weight[:])
            if self.num_update % 1000 == 0:
                print self.idx2name[index], weight.asnumpy()
        else:
            if state is not None:
                mom = state
                mom[:] *= self.momentum
                grad += wd * weight
                mom[:] += grad
                grad[:] += self.momentum * mom
                weight[:] += -lr * grad
            else:
                assert self.momentum == 0.0
                weight[:] += -lr * (grad + wd * weight)

    @staticmethod
    def soft_thresholding(input, alpha):
        return mx.ndarray.sign(input) * mx.ndarray.maximum(0.0, mx.ndarray.abs(input) - alpha)
