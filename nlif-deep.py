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

# os.environ['THEANO_FLAGS'] = 'device=gpu, floatX=float32'
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


def show_recons(x, z):
    plotting.compare([x.reshape(-1, 28, 28), z.reshape(-1, 28, 28)],
                     rows=5, cols=20, vlims=(-1, 2))


class Autoencoder(object):
    """Autoencoder with tied weights"""

    def __init__(self, vis_shape, n_hid,
                 W=None, c=None, b=None, mask=None,
                 rf_shape=None, hidlinear=False, vislinear=False, seed=22):
        dtype = theano.config.floatX

        self.vis_shape = vis_shape if isinstance(vis_shape, tuple) else (vis_shape,)
        self.n_vis = np.prod(vis_shape)
        self.n_hid = n_hid
        self.hidlinear = hidlinear
        self.vislinear = vislinear
        self.seed = seed

        rng = np.random.RandomState(seed=self.seed)
        self.theano_rng = theano.sandbox.rng_mrg.MRG_RandomStreams(seed=self.seed)

        # create initial weights and biases
        if W is None:
            Wmag = 4 * np.sqrt(6. / (self.n_vis + self.n_hid))
            W = rng.uniform(
                low=-Wmag, high=Wmag, size=(self.n_vis, self.n_hid)
            ).astype(dtype)

        if c is None:
            c = np.zeros(self.n_hid, dtype=dtype)

        if b is None:
            b = np.zeros(self.n_vis, dtype=dtype)

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
        W = W.astype(dtype)
        c = c.astype(dtype)
        b = b.astype(dtype)

        self.W = theano.shared(W, name='W')
        self.c = theano.shared(c, name='c')
        self.b = theano.shared(b, name='b')

    @classmethod
    def load(cls, filename):
        d = np.load(filename)['d'].item()
        return cls(**d)

    def save(self, filename):
        d = dict()
        for k, v in self.__dict__.items():
            if k in ['W', 'c', 'b']:
                d[k] = v.get_value()
            elif k in ['vis_shape', 'n_hid', 'rf_shape',
                       'mask', 'vislinear', 'hidlinear', 'seed']:
                d[k] = v
        np.savez(filename, d=d)

    @property
    def filters(self):
        if self.mask is None:
            return self.W.get_value().T.reshape((self.n_hid,) + self.vis_shape)
        else:
            filters = self.W.get_value().T[self.mask.T]
            shape = (self.n_hid,) + self.rf_shape
            return filters.reshape(shape)

    def propup(self, x):
        a = tt.dot(x, self.W) + self.c
        return a if self.hidlinear else nlif(a)

    def propdown(self, y):
        a = tt.dot(y, self.W.T) + self.b
        return a if self.vislinear else nlif(a)

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

    def auto_sgd(self, images, deep=None, test_images=None,
                     batch_size=100, rate=0.1, n_epochs=10):
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
        print batches.shape

        for epoch in range(n_epochs):
            costs = []
            for batch in batches:
                costs.append(train_dbn(batch))
                self.check_params()

            print "Epoch %d: %0.3f" % (epoch, np.mean(costs))

            if deep is not None and test_images is not None:
                # plot reconstructions on test set
                plt.figure(2)
                plt.clf()
                recons = deep.reconstruct(test_images)
                show_recons(test_images, recons)
                plt.draw()

            # plot filters for first layer only
            if deep is not None and self is deep.autos[0]:
                plt.figure(3)
                plt.clf()
                plotting.filters(self.filters, rows=10, cols=20)
                plt.draw()


class DeepAutoencoder(object):

    def __init__(self, autos=None):
        self.autos = autos if autos is not None else []
        self.W = None  # classifier weights
        self.b = None  # classifier biases

        self.seed = 90
        self.theano_rng = theano.sandbox.rng_mrg.MRG_RandomStreams(seed=self.seed)

    def propup(self, images):
        codes = images
        for auto in self.autos:
            codes = auto.propup(codes)
        return codes

    def propdown(self, codes):
        images = codes
        for auto in self.autos[::-1]:
            images = auto.propdown(images)
        return images

    @property
    def encode(self):
        images = tt.matrix('images')
        codes = self.propup(images)
        return theano.function([images], codes)

    @property
    def decode(self):
        codes = tt.matrix('codes')
        images = self.propdown(codes)
        return theano.function([codes], images)

    @property
    def reconstruct(self):
        x = tt.matrix('images')
        y = self.propup(x)
        z = self.propdown(y)
        return theano.function([x], z)

    def auto_sgd(self, images, test_images=None,
                 batch_size=100, rate=0.1, n_epochs=10):
        dtype = theano.config.floatX

        params = []
        for auto in self.autos:
            params.extend((auto.W, auto.c, auto.b))

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

        for auto in self.autos:
            if auto.mask is not None:
                updates[auto.W] = updates[auto.W] * auto.mask

        train_dbn = theano.function([x], error, updates=updates)

        # --- perform SGD
        batches = images.reshape(-1, batch_size, images.shape[1])
        assert np.isfinite(batches).all()
        print batches.shape

        for epoch in range(n_epochs):
            costs = []
            for batch in batches:
                costs.append(train_dbn(batch))
                # self.check_params()

            print "Epoch %d: %0.3f" % (epoch, np.mean(costs))

            if test_images is not None:
                # plot reconstructions on test set
                plt.figure(2)
                plt.clf()
                recons = deep.reconstruct(test_images)
                show_recons(test_images, recons)
                plt.draw()

            # plot filters for first layer only
            plt.figure(3)
            plt.clf()
            plotting.filters(self.autos[0].filters, rows=10, cols=20)
            plt.draw()

    def train_classifier(self, train, test):

        dtype = theano.config.floatX

        # --- find codes
        images, labels = train
        n_labels = len(np.unique(labels))
        codes = self.encode(images.astype(dtype))

        codes = theano.shared(codes.astype(dtype), name='codes')
        labels = tt.cast(theano.shared(labels.astype(dtype), name='labels'), 'int32')

        # --- compute backprop function
        Wshape = (self.autos[-1].n_hid, n_labels)
        x = tt.matrix('x', dtype=dtype)
        y = tt.ivector('y')
        W = tt.matrix('W', dtype=dtype)
        b = tt.vector('b', dtype=dtype)

        W0 = np.random.normal(size=Wshape).astype(dtype).flatten() / 10
        b0 = np.zeros(n_labels)

        split_p = lambda p: [p[:-n_labels].reshape(Wshape), p[-n_labels:]]
        form_p = lambda params: np.hstack([p.flatten() for p in params])

        # compute negative log likelihood
        p_y_given_x = tt.nnet.softmax(tt.dot(x, W) + b)
        y_pred = tt.argmax(p_y_given_x, axis=1)
        nll = -tt.mean(tt.log(p_y_given_x)[tt.arange(y.shape[0]), y])
        error = tt.mean(tt.neq(y_pred, y))

        # compute gradients
        grads = tt.grad(nll, [W, b])
        f_df = theano.function(
            [W, b], [error] + grads,
            givens={x: codes, y: labels})

        # --- begin backprop
        def f_df_wrapper(p):
            w, b = split_p(p)
            outs = f_df(w.astype(dtype), b.astype(dtype))
            cost, grad = outs[0], form_p(outs[1:])
            return cost.astype('float64'), grad.astype('float64')

        p0 = form_p([W0, b0])
        p_opt, mincost, info = scipy.optimize.lbfgsb.fmin_l_bfgs_b(
            f_df_wrapper, p0, maxfun=100, iprint=1)

        self.W, self.b = split_p(p_opt)

    def backprop(self, train_set, test_set, n_epochs=30):
        dtype = theano.config.floatX

        params = []
        for auto in self.autos:
            params.extend([auto.W, auto.c])

        # --- compute backprop function
        assert self.W is not None and self.b is not None
        W = theano.shared(self.W.astype(dtype), name='Wc')
        b = theano.shared(self.b.astype(dtype), name='bc')

        x = tt.matrix('batch')
        y = tt.ivector('labels')

        # compute coding error
        p_y_given_x = tt.nnet.softmax(tt.dot(self.propup(x), W) + b)
        y_pred = tt.argmax(p_y_given_x, axis=1)
        nll = -tt.mean(tt.log(p_y_given_x)[tt.arange(y.shape[0]), y])
        error = tt.mean(tt.neq(y_pred, y))

        # compute gradients
        grads = tt.grad(nll, params)
        f_df = theano.function([x, y], [error] + grads)

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

        # --- run L_BFGS
        images, labels = train
        labels = labels.astype('int32')

        def f_df_wrapper(p):
            for param, value in zip(params, split_p(p)):
                param.set_value(value.astype(param.dtype))

            outs = f_df(images, labels)
            cost, grads = outs[0], outs[1:]
            grad = form_p(grads)
            return cost.astype('float64'), grad.astype('float64')

        p0 = form_p(np_params)
        p_opt, mincost, info = scipy.optimize.lbfgsb.fmin_l_bfgs_b(
            f_df_wrapper, p0, maxfun=100, iprint=1)

        for param, value in zip(params, split_p(p_opt)):
            param.set_value(value.astype(param.dtype), borrow=False)

    def test(self, test_set):
        assert self.W is not None and self.b is not None

        images, labels = test_set
        codes = self.encode(images)

        categories = np.unique(labels)
        inds = np.argmax(np.dot(codes, self.W) + self.b, axis=1)
        return (labels != categories[inds])


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

# --- pretrain with SGD backprop
shapes = [(28, 28), 200, 50]
linear = [True, False, True]
rf_shapes = [(9, 9), None]
rates = [1., 0.3]
n_layers = len(shapes) - 1
assert len(linear) == len(shapes)
assert len(rf_shapes) == n_layers
assert len(rates) == n_layers

n_epochs = 15
batch_size = 100

deep = DeepAutoencoder()
data = train_images
for i in range(n_layers):
    savename = "nlif-deep-%d.npz" % i
    if not os.path.exists(savename):
        auto = Autoencoder(
            shapes[i], shapes[i+1], rf_shape=rf_shapes[i],
            vislinear=linear[i], hidlinear=linear[i+1])
        deep.autos.append(auto)
        auto.auto_sgd(data, deep, test_images,
                      n_epochs=n_epochs, rate=rates[i])
        auto.save(savename)
    else:
        auto = Autoencoder.load(savename)
        deep.autos.append(auto)

    data = auto.encode(data)

plt.figure(99)
plt.clf()
recons = deep.reconstruct(test_images)
show_recons(test_images, recons)

print "recons error", rms(test_images - recons, axis=1).mean()

# deep.auto_sgd(train_images, test_images, rate=0.3, n_epochs=30)

print "recons error", rms(test_images - recons, axis=1).mean()

# print "mean error", deep.test(test).mean()

# --- train classifier with backprop
savename = 'classifier.npz'
if not os.path.exists(savename):
    deep.train_classifier(train, test)
    np.savez(savename, W=deep.W, b=deep.b)
else:
    savedata = np.load(savename)
    deep.W, deep.b = savedata['W'], savedata['b']

print "mean error", deep.test(test).mean()

# --- train with backprop
deep.backprop(train, test, n_epochs=100)

print "mean error", deep.test(test).mean()
