
import os
import warnings

import numpy as np

# os.environ['THEANO_FLAGS'] = 'device=gpu, floatX=float32'
# os.environ['THEANO_FLAGS'] = 'exception_verbosity=high'
import theano
import theano.tensor as tt
from theano.tensor.signal.downsample import max_pool_2d


def average_pool_2d(x, ds):
    n_batch, n_ch, s0, s1 = x.shape
    d0, d1 = ds
    c = tt.ones((s0, s1))

    # sum elements in regions
    y = x[:, :, 0::d0, 0::d1].copy()
    d = c[0::d0, 0::d1].copy()
    for i in range(0, d0):
        for j in range(0, d1):
            if i != 0 or j != 0:
                ni = (s0 - i - 1) / d0 + 1
                nj = (s1 - j - 1) / d1 + 1
                # tt.inc_subtensor(y[:, :, :ni, :nj], x[:, :, i::d0, j::d1], inplace=True)
                # tt.inc_subtensor(d[:ni, :nj], c[i::d0, j::d1], inplace=True)
                y = tt.inc_subtensor(y[:, :, :ni, :nj], x[:, :, i::d0, j::d1])
                d = tt.inc_subtensor(d[:ni, :nj], c[i::d0, j::d1])

    # divide by number of elements
    y /= d

    return y


def power_pool_2d(x, ds, p=3, b=0):
    n_batch, n_ch, s0, s1 = x.shape
    d0, d1 = ds
    c = tt.ones((s0, s1))

    # sum elements in regions
    y = tt.abs_(x[:, :, 0::d0, 0::d1])**p
    d = c[0::d0, 0::d1].copy()
    for i in range(0, d0):
        for j in range(0, d1):
            if i != 0 or j != 0:
                ni = (s0 - i - 1) / d0 + 1
                nj = (s1 - j - 1) / d1 + 1
                xij = tt.abs_(x[:, :, i::d0, j::d1])**p
                y = tt.inc_subtensor(y[:, :, :ni, :nj], xij)
                d = tt.inc_subtensor(d[:ni, :nj], c[i::d0, j::d1])

    # divide by number of elements
    y /= d
    y += b**p

    # take root
    y = y**(1. / p)

    return y


def test_pool_2d():
    import numpy as np
    import theano

    # s = (6, 6)
    # s = (100, 32, 6, 6)
    s = (1, 1, 3, 3)
    p = (2, 2)

    x = tt.tensor4()
    y = average_pool_2d(x, p)
    # y = average_pool_b(x, p)
    # y = power_pool_2d(x, p, p=3)
    f = theano.function([x], y)

    # x = np.arange(12).reshape(1, 1, 3, 4)
    # x = np.arange(16).reshape(1, 1, 4, 4)
    x = np.arange(np.prod(s)).reshape(s)

    y = f(x.astype(theano.config.floatX))
    print x[0, 0]
    print y
    # print theano.printing.debugprint(f)


if __name__ == '__main__':
    test_pool_2d()
