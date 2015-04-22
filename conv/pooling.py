
import os
import warnings

import numpy as np

os.environ['THEANO_FLAGS'] = 'device=gpu, floatX=float32'
# os.environ['THEANO_FLAGS'] = 'exception_verbosity=high'
import theano
import theano.tensor as tt
from theano.tensor.signal.downsample import max_pool_2d


def average_pool_2d(input, ds):

    n_batch, n_ch, s0, s1 = input.shape
    d0, d1 = ds

    # pad input to be divisible into regions
    p0 = tt.cast(tt.ceil(s0 / float(d0)) * d0, dtype=s0.dtype)
    p1 = tt.cast(tt.ceil(s1 / float(d1)) * d1, dtype=s1.dtype)

    x = input
    x = tt.concatenate([x, tt.zeros((n_batch, n_ch, p0 - s0, s1))], axis=2)
    x = tt.concatenate([x, tt.zeros((n_batch, n_ch, p0, p1 - s1))], axis=3)

    c = tt.ones((s0, s1))
    c = tt.concatenate([c, tt.zeros((p0 - s0, s1))], axis=0)
    c = tt.concatenate([c, tt.zeros((p0, p1 - s1))], axis=1)

    # sum elements in regions
    y = x[:, :, 0::d0, 0::d1].copy()
    d = c[0::d0, 0::d1].copy()
    for i in range(0, d0):
        for j in range(0, d1):
            if i != 0 or j != 0:
                y += x[:, :, i::d0, j::d1]
                d += c[i::d0, j::d1]

    # divide by number of elements
    y /= d

    return y


def average_pool_b(input, ds):

    n_batch, n_ch, s0, s1 = input.shape
    d0, d1 = ds

    # pad input to be divisible into regions
    # p0 = tt.cast(tt.ceil(s0 / float(d0)) * d0, dtype=s0.dtype)
    # p1 = tt.cast(tt.ceil(s1 / float(d1)) * d1, dtype=s1.dtype)

    x = input
    # x = tt.concatenate([x, tt.zeros((n_batch, n_ch, p0 - s0, s1))], axis=2)
    # x = tt.concatenate([x, tt.zeros((n_batch, n_ch, p0, p1 - s1))], axis=3)

    c = tt.ones((s0, s1))
    # c = tt.concatenate([c, tt.zeros((p0 - s0, s1))], axis=0)
    # c = tt.concatenate([c, tt.zeros((p0, p1 - s1))], axis=1)

    # sum elements in regions
    y = x[:, :, 0::d0, 0::d1].copy()
    d = c[0::d0, 0::d1].copy()
    for i in range(0, d0):
        for j in range(0, d1):
            if i != 0 or j != 0:
                y += x[:, :, i::d0, j::d1]
                d += c[i::d0, j::d1]

    # divide by number of elements
    y /= d

    return y


def power_pool_2d(input, ds, p=3, b=0):

    n_batch, n_ch, s0, s1 = input.shape
    d0, d1 = ds

    # pad input to be divisible into regions
    p0 = tt.cast(tt.ceil(s0 / float(d0)) * d0, dtype=s0.dtype)
    p1 = tt.cast(tt.ceil(s1 / float(d1)) * d1, dtype=s1.dtype)

    x = input
    x = tt.concatenate([x, tt.zeros((n_batch, n_ch, p0 - s0, s1))], axis=2)
    x = tt.concatenate([x, tt.zeros((n_batch, n_ch, p0, p1 - s1))], axis=3)

    c = tt.ones((s0, s1))
    c = tt.concatenate([c, tt.zeros((p0 - s0, s1))], axis=0)
    c = tt.concatenate([c, tt.zeros((p0, p1 - s1))], axis=1)

    # sum elements in regions
    y = tt.abs_(x[:, :, 0::d0, 0::d1])**p
    d = c[0::d0, 0::d1].copy()
    for i in range(0, d0):
        for j in range(0, d1):
            if i != 0 or j != 0:
                y += tt.abs_(x[:, :, i::d0, j::d1])**p
                d += c[i::d0, j::d1]

    # divide by number of elements
    y += b
    y /= d

    # take root
    y = y**(1. / p)

    return y


def test_pool_2d():
    import numpy as np
    import theano

    # s = (6, 6)
    # s = (100, 32, 6, 6)
    s = (1, 1, 6, 6)
    p = (3, 3)

    x = tt.tensor4()
    # y = average_pool_2d(x, p)
    y = average_pool_b(x, p)
    # y = power_pool_2d(x, p, p=3)
    f = theano.function([x], y)

    # x = np.arange(12).reshape(1, 1, 3, 4)
    # x = np.arange(16).reshape(1, 1, 4, 4)
    x = np.arange(np.prod(s)).reshape(s)

    y = f(x.astype(theano.config.floatX))
    print x[0, 0]
    print y


if __name__ == '__main__':
    test_pool_2d()
