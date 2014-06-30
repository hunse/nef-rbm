"""
Training an autoencoder with LIF-likes
"""

import collections
import os
import gzip
import cPickle as pickle
import urllib

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

os.environ['THEANO_FLAGS'] = 'device=gpu, floatX=float32'
# os.environ['THEANO_FLAGS'] = 'mode=DEBUG_MODE'
import theano
import theano.tensor as tt
import theano.sandbox.rng_mrg

import plotting

plt.ion()


def norm(x, **kwargs):
    return np.sqrt((x**2).sum(**kwargs))


def rms(x, **kwargs):
    return np.sqrt((x**2).mean(**kwargs))


def nlif(x):
    dtype = theano.config.floatX
    sigma = tt.cast(0.05, dtype=dtype)
    tau_ref = tt.cast(0.002, dtype=dtype)
    tau_rc = tt.cast(0.02, dtype=dtype)
    alpha = tt.cast(1, dtype=dtype)
    beta = tt.cast(1, dtype=dtype)
    amp = tt.cast(1. / 65, dtype=dtype)

    j = alpha * x + beta - 1
    j = sigma * tt.log1p(tt.exp(j / sigma))
    v = amp / (tau_ref + tau_rc * tt.log1p(1. / j))
    return tt.switch(j > 0, v, 0.0)


class Autoencoder(object):
    """Autoencoder with tied weights"""

    def __init__(self, vis_shape, n_hid,
                 W=None, c=None, b=None, mask=None,
                 rf_shape=None, seed=22):
        self.dtype = theano.config.floatX

        self.vis_shape = vis_shape if isinstance(vis_shape, tuple) else (vis_shape,)
        self.n_vis = np.prod(vis_shape)
        self.n_hid = n_hid
        self.seed = seed

        self.nonlinearity = nlif

        rng = np.random.RandomState(seed=self.seed)
        self.theano_rng = theano.sandbox.rng_mrg.MRG_RandomStreams(seed=self.seed)

        # create initial weights and biases
        if W is None:
            Wmag = 4 * np.sqrt(6. / (self.n_vis + self.n_hid))
            W = rng.uniform(
                low=-Wmag, high=Wmag, size=(self.n_vis, self.n_hid)
            ).astype(self.dtype)

        if c is None:
            c = np.zeros(self.n_hid, dtype=self.dtype)

        if b is None:
            b = np.zeros(self.n_vis, dtype=self.dtype)

        # create initial sparsity mask
        self.rf_shape = rf_shape
        self.mask = mask
        if rf_shape is not None and mask is None:
            assert isinstance(vis_shape, tuple) and len(vis_shape) == 2
            M, N = vis_shape
            m, n = rf_shape

            # find random positions for top-left corner of each RF
            i = rng.randint(low=0, high=M-m+1, size=self.n_hid)
            j = rng.randint(low=0, high=N-n+1, size=self.n_hid)

            mask = np.zeros((M, N, self.n_hid), dtype='bool')
            for k in xrange(self.n_hid):
                mask[i[k]:i[k]+m, j[k]:j[k]+n, k] = True

            self.mask = mask.reshape(self.n_vis, self.n_hid)
            W = W * self.mask  # make initial W sparse

        # create states for weights and biases
        W = W.astype(self.dtype)
        c = c.astype(self.dtype)
        b = b.astype(self.dtype)

        self.W = theano.shared(W, name='W')
        self.c = theano.shared(c, name='c')
        self.b = theano.shared(b, name='b')

    # @classmethod
    # def load(cls, filename):
    #     with open(filename, 'rb') as f:
    #         obj = pickle.load(f)
    #     return obj

    # def save(self, filename):
    #     with open(filename, 'wb') as f:
    #         pickle.dump(self, f)

    @property
    def filters(self):
        if self.mask is None:
            return self.W.get_value().T.reshape((self.n_hid,) + self.vis_shape)
        else:
            filters = self.W.get_value().T[self.mask.T]
            shape = (self.n_hid,) + self.rf_shape
            return filters.reshape(shape)

    def propup(self, x):
        return self.nonlinearity(tt.dot(x, self.W) + self.c)

    def propdown(self, y):
        return tt.dot(y, self.W.T) + self.b

    @property
    def encode(self):
        data = tt.matrix('data')
        code = self.propup(data)
        return theano.function([data], code)

    @property
    def decode(self):
        code = tt.matrix('code')
        data = self.propdown(code)
        return theano.function([code], data)

    def check_params(self):
        for param in [self.W, self.c, self.b]:
            if param is not None:
                assert np.isfinite(param.get_value()).all()

    def sgd_backprop(self, images, test_images, batch_size=100, rate=0.1, n_epochs=10):
        dtype = theano.config.floatX

        params = [self.W, self.c, self.b]

        # --- compute backprop function
        x = tt.matrix('images')
        xn = x + self.theano_rng.normal(size=x.shape, std=1, dtype=dtype)

        # compute coding error
        y = self.propup(xn)
        z = self.propdown(y)
        rmses = tt.sqrt(tt.mean((x - z)**2, axis=1))
        error = tt.mean(rmses)

        # compute gradients
        grads = tt.grad(error, params)
        updates = collections.OrderedDict()
        for param, grad in zip(params, grads):
            updates[param] = param - tt.cast(rate, dtype) * grad

        if self.mask is not None:
            updates[self.W] = updates[self.W] * self.mask

        train_dbn = theano.function([x], error, updates=updates)

        # --- perform SGD
        batches = images.reshape(-1, batch_size, images.shape[1])
        assert np.isfinite(batches).all()

        for epoch in range(n_epochs):
            costs = []
            for batch in batches:
                costs.append(train_dbn(batch))
                self.check_params()

            print "Epoch %d: %0.3f" % (epoch, np.mean(costs))

            # plot reconstructions on test set
            plt.figure(2)
            plt.clf()
            x = test_images
            y = self.encode(test_images)
            z = self.decode(y)
            plotting.compare(
                [x.reshape(-1, 28, 28), z.reshape(-1, 28, 28)],
                rows=5, cols=20, vlims=(-1, 2))
            plt.draw()

            print "Test error:", rms(x - z, axis=1).mean()

            # plot filters for first layer only
            plt.figure(3)
            plt.clf()
            plotting.filters(self.filters, rows=10, cols=20)
            plt.draw()


# --- load the data
filename = 'mnist.pkl.gz'

if not os.path.exists(filename):
    url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
    urllib.urlretrieve(url, filename=filename)

with gzip.open(filename, 'rb') as f:
    train, valid, test = pickle.load(f)

train_images, _ = train
valid_images, _ = valid
test_images, _ = test

for images in [train_images, valid_images, test_images]:
    images -= images.mean(axis=0, keepdims=True)
    images /= np.maximum(images.std(axis=0, keepdims=True), 3e-1)

# --- pretrain with CD
# n_hid = 500
n_hid = 200
# n_hid = 150

n_epochs = 15
rate = 1
batch_size = 100
# batch_size = 20

auto = Autoencoder((28, 28), n_hid, rf_shape=(9, 9))

auto.sgd_backprop(train_images, test_images, n_epochs=n_epochs, rate=rate)
