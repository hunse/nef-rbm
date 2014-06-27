"""
Training a simple RBM with ReLUs
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


class RBM(object):

    # --- define RBM parameters
    def __init__(self, vis_shape, n_hid,
                 W=None, c=None, b=None, mask=None,
                 rf_shape=None, seed=22):
        self.dtype = theano.config.floatX

        self.vis_shape = vis_shape if isinstance(vis_shape, tuple) else (vis_shape,)
        self.n_vis = np.prod(vis_shape)
        self.n_hid = n_hid
        # self.gaussian = gaussian
        # self.hidlinear = hidlinear
        self.seed = seed

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

        # create states for initial increments (for momentum)
        self.Winc = theano.shared(np.zeros_like(W), name='Winc')
        self.cinc = theano.shared(np.zeros_like(c), name='cinc')
        self.binc = theano.shared(np.zeros_like(b), name='binc')

    def save(self, filename):
        d = dict()
        for k, v in self.__dict__.items():
            if k in ['W', 'c', 'b']:
                d[k] = v.get_value()
            elif k in ['vis_shape', 'n_hid', 'rf_shape',
                       'mask', 'hidlinear', 'seed']:
                d[k] = v
        np.savez(filename, dict=d)

    @classmethod
    def load(cls, filename):
        d = np.load(filename)['dict'].item()
        return cls(**d)

    @property
    def filters(self):
        if self.mask is None:
            return self.W.get_value().T.reshape((self.n_hid,) + self.vis_shape)
        else:
            filters = self.W.get_value().T[self.mask.T]
            shape = (self.n_hid,) + self.rf_shape
            return filters.reshape(shape)

    # --- define RBM propagation functions
    def HgivenV(self, vis):
        a = tt.dot(vis, self.W) + self.c
        return a, tt.nnet.sigmoid(a)
        # return a, tt.matt.nnet.sigmoid(a)

    def VgivenH(self, hid):
        a = tt.dot(hid, self.W.T) + self.b
        # return a, tt.nnet.sigmoid(a)
        return a, a

    def sampHgivenV(self, vis):
        _, hidprob = self.HgivenV(vis)
        hidsamp = self.theano_rng.binomial(
            size=hidprob.shape, n=1, p=hidprob, dtype=self.dtype)
        return hidprob, hidsamp

    # --- define RBM updates
    def get_cost_updates(self, data, rate=0.1, weightcost=2e-4, momentum=0.5):

        numcases = tt.cast(data.shape[0], self.dtype)
        rate = tt.cast(rate, self.dtype)
        weightcost = tt.cast(weightcost, self.dtype)
        momentum = tt.cast(momentum, self.dtype)

        # compute positive phase
        poshidprob, poshidsamp = self.sampHgivenV(data)

        posw = tt.dot(data.T, poshidprob) / numcases
        posb = tt.mean(data, axis=0)
        # posb = tt.mean(data - self.b, axis=0)
        posc = tt.mean(poshidprob, axis=0)

        # compute negative phase
        _, negdata = self.VgivenH(poshidsamp)
        _, neghidprob = self.HgivenV(negdata)
        negw = tt.dot(negdata.T, neghidprob) / numcases
        negb = tt.mean(negdata, axis=0)
        # negb = tt.mean(negdata - self.b, axis=0)
        negc = tt.mean(neghidprob, axis=0)

        # compute error
        rmse = tt.sqrt(tt.mean((data - negdata)**2, axis=1))
        err = tt.mean(rmse)

        # compute updates
        Winc = momentum * self.Winc + rate * (posw - negw - weightcost * self.W)
        binc = momentum * self.binc + rate * (posb - negb)
        cinc = momentum * self.cinc + rate * (posc - negc)

        if self.mask is not None:
            Winc = Winc * self.mask

        updates = [
            (self.W, self.W + Winc),
            (self.c, self.c + cinc),
            (self.b, self.b + binc),
            (self.Winc, Winc),
            (self.cinc, cinc),
            (self.binc, binc)
        ]

        return err, updates

    @property
    def encode(self):
        data = tt.matrix('data', dtype=self.dtype)
        _, code = self.HgivenV(data)
        return theano.function([data], code)

    @property
    def decode(self):
        codes = tt.matrix('codes', dtype=self.dtype)
        _, data = self.VgivenH(codes)
        return theano.function([codes], data)

    def pretrain(self, batches, test_images, n_epochs=10, **train_params):

        data = tt.matrix('data', dtype=self.dtype)
        cost, updates = self.get_cost_updates(data, **train_params)
        train_rbm = theano.function([data], cost, updates=updates)

        for epoch in range(n_epochs):

            # train on each mini-batch
            costs = []
            for batch in batches:
                costs.append(train_rbm(batch))

            print "Epoch %d: %0.3f" % (epoch, np.mean(costs))

            # plot reconstructions on test set
            plt.figure(2)
            plt.clf()
            x = test_images
            y = rbm.encode(test_images)
            z = rbm.decode(y)
            plotting.compare(
                [x.reshape(-1, 28, 28), z.reshape(-1, 28, 28)],
                rows=5, cols=20)
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
    # images -= images.mean()
    images -= images.mean(axis=0, keepdims=True)
    images /= np.maximum(images.std(axis=0, keepdims=True), 3e-1)

plt.figure(1)
plt.clf()
plt.imshow(train_images[0].reshape(28, 28), cmap='gray')
# assert False

# --- pretrain with CD
n_hid = 500

n_epochs = 15
# rate = 0.1
rate = 0.05
# rate = 0.01
batch_size = 100

batches = train_images.reshape(-1, batch_size, train_images.shape[1])

rbm = RBM((28, 28), n_hid, rf_shape=(9, 9))

rbm.pretrain(batches, test_images, n_epochs=n_epochs, rate=rate)

# plt.figure(99)
# plt.clf()

# recons = dbn.reconstruct(test_images)
# plotting.compare([test_images.reshape(-1, 28, 28),
#                   recons.reshape(-1, 28, 28)],
#                  rows=5, cols=20)

# print "mean error", dbn.test(train, test).mean()
