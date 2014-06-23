"""
Train an MNIST RBM, based off the demo code at
    http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html
"""

import collections
import os
import gzip
import cPickle as pickle
import urllib

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

# os.environ['THEANO_FLAGS'] = 'device=gpu, floatX=float32'
# os.environ['THEANO_FLAGS'] = 'mode=DEBUG_MODE'
import theano
import theano.tensor as tt
import theano.sandbox.rng_mrg

import nengo
import nengo.utils.distributions as dists

import plotting

plt.ion()


def norm(x, **kwargs):
    return np.sqrt((x**2).sum(**kwargs))


class RBM(object):

    # --- define RBM parameters
    def __init__(self, vis_shape, n_hid,
                 encoders=None,
                 # intercepts=dists.Uniform(-1, 1),
                 intercepts=dists.Uniform(-0.5, -0.5),
                 # max_rates=dists.Uniform(150, 250),
                 # max_rates=dists.Uniform(1, 1),
                 max_rate=200,
                 neurons=nengo.LIF(),
                 mask=None, rf_shape=None, seed=None):



        if seed is None:
            seed = np.random.randint(2**31 - 1)

        vis_shape = vis_shape if isinstance(vis_shape, tuple) else (vis_shape,)
        n_vis = np.prod(vis_shape)

        rng = np.random.RandomState(seed=seed)

        # create initial parameters
        if encoders is None:
            encoders = rng.normal(size=(n_hid, n_vis))

        if isinstance(intercepts, dists.Distribution):
            intercepts = intercepts.sample(n_hid, rng=rng)

        # if isinstance(max_rates, dists.Distribution):
        #     max_rates = max_rates.sample(n_hid, rng=rng)
        max_rates = max_rate * np.ones(n_hid)

        # create initial sparsity mask
        if rf_shape is not None and mask is None:
            assert isinstance(vis_shape, tuple) and len(vis_shape) == 2
            M, N = vis_shape
            m, n = rf_shape

            # find random positions for top-left corner of each RF
            i = rng.randint(low=0, high=M-m+1, size=n_hid)
            j = rng.randint(low=0, high=N-n+1, size=n_hid)

            mask = np.zeros((n_hid, M, N), dtype='bool')
            for k in xrange(n_hid):
                mask[k, i[k]:i[k]+m, j[k]:j[k]+n] = True

            mask = mask.reshape(n_hid, n_vis)

        if mask is not None:
            encoders = encoders * mask
        encoders /= norm(encoders, axis=1, keepdims=True)

        gain, bias = neurons.gain_bias(max_rates, intercepts)

        self.vis_shape = vis_shape
        self.n_vis = n_vis
        self.n_hid = n_hid
        self.rf_shape = rf_shape
        self.seed = seed

        self.neurons = neurons
        self.encoders = encoders
        self.max_rates = max_rates
        self.gain = gain
        self.bias = bias
        self.mask = mask
        self.decoders = None

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
        return obj

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def encode(self, x):
        e = np.dot(x, self.encoders.T)
        return self.neurons.rates(e, self.gain, self.bias) / self.max_rates

    def decode(self, y):
        assert self.decoders is not None
        return np.dot(y, self.decoders)

    # def pretrain(self, batches, dbn=None, test_images=None,
    #              n_epochs=10, **train_params):
    def pretrain(self, images):
        acts = self.encode(images)
        solver = nengo.decoders.LstsqL2()
        self.decoders, info = solver(acts, images)
        print "Trained RBM: %0.3f" % (info['rmses'].mean())


class DBN(object):

    def __init__(self, rbms=None):
        self.dtype = theano.config.floatX
        self.rbms = rbms if rbms is not None else []
        self.W = None  # classifier weights
        self.b = None  # classifier biases

    # def propup(self, images):
    #     codes = images
    #     for rbm in self.rbms:
    #         codes = rbm.probHgivenV(codes)
    #     return codes

    # def propdown(self, codes):
    #     images = codes
    #     for rbm in self.rbms[::-1]:
    #         images = rbm.probVgivenH(images)
    #     return images

    # @property
    # def encode(self):
    #     images = tt.matrix('images', dtype=self.dtype)
    #     codes = self.propup(images)
    #     return theano.function([images], codes)

    # @property
    # def decode(self):
    #     codes = tt.matrix('codes', dtype=self.dtype)
    #     images = self.propdown(codes)
    #     return theano.function([codes], images)

    def encode(self, x):
        for rbm in self.rbms:
            x = rbm.encode(x)
        return x

    def decode(self, x):
        for rbm in self.rbms[::-1]:
            x = rbm.decode(x)
        return x

    def reconstruct(self, x):
        y = self.encode(x)
        z = self.decode(y)
        return z

    def backprop_autoencoder(self, train_set):
        dtype = self.rbms[0].dtype
        params = []
        for rbm in self.rbms:
            params.extend([rbm.W, rbm.c])

        # --- compute backprop function
        x = tt.matrix('batch', dtype=dtype)

        # compute coding error
        y = self.propdown(self.propup(x))
        rmses = tt.sqrt(tt.mean((x - y)**2, axis=1))
        error = tt.mean(rmses)

        # compute gradients
        grads = tt.grad(error, params)
        f_df = theano.function([x], [error] + grads)

        np_params = [param.get_value() for param in params]
        def split_p(p):
            split = []
            i = 0
            for param in np_params:
                split.append(p[i:i + param.size].reshape(param.shape))
                i += param.size
            return split

        def form_p(params):
            return np.hstack([param.flatten() for param in params])

        # --- find target codes
        images, _ = train

        def f_df_wrapper(p):
            for param, value in zip(params, split_p(p)):
                param.set_value(value.astype(param.dtype))

            outs = f_df(images)
            cost, grads = outs[0], outs[1:]
            grad = form_p(grads)
            return cost.astype('float64'), grad.astype('float64')

        p0 = form_p(np_params)
        p_opt, mincost, info = scipy.optimize.lbfgsb.fmin_l_bfgs_b(
            f_df_wrapper, p0, maxfun=100, iprint=1)

        for param, value in zip(params, split_p(p_opt)):
            param.set_value(value.astype(param.dtype), borrow=False)

    def test_reconstruction(self, images):
        recons = self.reconstruct(images)
        rmses = np.sqrt(np.mean((images - recons)**2, axis=1))
        return rmses


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
    images[:] = 2 * images - 1

# --- pretrain with CD
shapes = [(28, 28), 500, 200, 50]
n_layers = len(shapes) - 1
rf_shapes = [(9, 9), None, None]
assert len(rf_shapes) == n_layers

dbn = DBN()
# data = train_images[:1000]
data = train_images[:10000]
valid_images = valid_images[:1000]
for i in range(n_layers):
    # savename = "pretrained_rbm_%d.npz" % i
    # if not os.path.exists(savename):
    # batches = data.reshape(
    #     data.shape[0] / batch_size, batch_size, data.shape[1])

    rbm = RBM(shapes[i], shapes[i+1], rf_shape=rf_shapes[i])
    rbm.pretrain(data)
    # rbm.pretrain(batches, dbn, valid_ima,
    #              n_epochs=n_epochs, rate=rates[i])
        # rbm.save(savename)
    # else:
    #     rbm = RBM.load(savename)
    #     dbn.rbms.append(rbm)

    data = rbm.encode(data)

    dbn.rbms.append(rbm)
    rmses = dbn.test_reconstruction(valid_images)
    print "RBM %d error: %0.3f (%0.3f)" % (i, rmses.mean(), rmses.std())


plt.figure(99)
plt.clf()
recons = dbn.reconstruct(test_images)
plotting.compare(
    [test_images.reshape(-1, 28, 28), recons.reshape(-1, 28, 28)],
    rows=5, cols=20)
plt.show()
