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

import plotting

plt.ion()


def norm(x, **kwargs):
    return np.sqrt((x**2).sum(**kwargs))


class RBM(object):

    # --- define RBM parameters
    def __init__(self, vis_shape, n_hid,
                 W=None, c=None, b=None, mask=None,
                 rf_shape=None, hidlinear=False, seed=22):
        self.dtype = theano.config.floatX

        self.vis_shape = vis_shape if isinstance(vis_shape, tuple) else (vis_shape,)
        self.n_vis = np.prod(vis_shape)
        self.n_hid = n_hid
        # self.gaussian = gaussian
        self.hidlinear = hidlinear
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
    def probHgivenV(self, vis):
        x = tt.dot(vis, self.W) + self.c
        if self.hidlinear:
            return x
        else:
            return tt.nnet.sigmoid(x)

    def probVgivenH(self, hid):
        x = tt.dot(hid, self.W.T) + self.b
        return tt.nnet.sigmoid(x)

    def sampHgivenV(self, vis):
        hidprob = self.probHgivenV(vis)
        if self.hidlinear:
            hidsamp = hidprob + self.theano_rng.normal(
                size=hidprob.shape, dtype=self.dtype)
        else:
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

        posprods = tt.dot(data.T, poshidprob) / numcases
        posvisact = tt.mean(data, axis=0)
        poshidact = tt.mean(poshidprob, axis=0)

        # compute negative phase
        negdata = self.probVgivenH(poshidsamp)
        neghidprob = self.probHgivenV(negdata)
        negprods = tt.dot(negdata.T, neghidprob) / numcases
        negvisact = tt.mean(negdata, axis=0)
        neghidact = tt.mean(neghidprob, axis=0)

        # compute error
        rmse = tt.sqrt(tt.mean((data - negdata)**2, axis=1))
        err = tt.mean(rmse)

        # compute updates
        Winc = momentum * self.Winc + rate * (
            (posprods - negprods) - weightcost * self.W)
        cinc = momentum * self.cinc + rate * (poshidact - neghidact)
        binc = momentum * self.binc + rate * (posvisact - negvisact)

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
        code = self.probHgivenV(data)
        return theano.function([data], code)

    def pretrain(self, batches, dbn=None, test_images=None,
                 n_epochs=10, **train_params):

        data = tt.matrix('data', dtype=self.dtype)
        cost, updates = self.get_cost_updates(data, **train_params)
        train_rbm = theano.function([data], cost, updates=updates)

        for epoch in range(n_epochs):

            # train on each mini-batch
            costs = []
            for batch in batches:
                costs.append(train_rbm(batch))

            print "Epoch %d: %0.3f" % (epoch, np.mean(costs))

            if dbn is not None and test_images is not None:
                # plot reconstructions on test set
                plt.figure(2)
                plt.clf()
                recons = dbn.reconstruct(test_images)
                plotting.compare([test_images.reshape(-1, 28, 28),
                                  recons.reshape(-1, 28, 28)],
                                 rows=5, cols=20)
                plt.draw()

            # plot filters for first layer only
            if dbn is not None and self is dbn.rbms[0]:
                plt.figure(3)
                plt.clf()
                plotting.filters(self.filters, rows=10, cols=20)
                plt.draw()


class DBN(object):

    def __init__(self, rbms=None):
        self.dtype = theano.config.floatX
        self.rbms = rbms if rbms is not None else []
        self.W = None  # classifier weights
        self.b = None  # classifier biases

        self.theano_rng = theano.sandbox.rng_mrg.MRG_RandomStreams(seed=90)

    def propup(self, images):
        codes = images
        for rbm in self.rbms:
            codes = rbm.probHgivenV(codes)
        return codes

    def propdown(self, codes):
        images = codes
        for rbm in self.rbms[::-1]:
            images = rbm.probVgivenH(images)
        return images

    @property
    def encode(self):
        images = tt.matrix('images', dtype=self.dtype)
        codes = self.propup(images)
        return theano.function([images], codes)

    @property
    def decode(self):
        codes = tt.matrix('codes', dtype=self.dtype)
        images = self.propdown(codes)
        return theano.function([codes], images)

    @property
    def reconstruct(self):
        images = tt.matrix('images', dtype=self.dtype)
        codes = self.propup(images)
        recons = self.propdown(codes)
        return theano.function([images], recons)

    def get_categories_vocab(self, train_set, normalize=True):
        images, labels = train_set

        # find mean codes for each label
        codes = dbn.encode(images)
        categories = np.unique(labels)
        vocab = []
        for category in categories:
            pointer = codes[labels == category].mean(0)
            vocab.append(pointer)

        vocab = np.array(vocab, dtype=codes.dtype)
        if normalize:
            vocab /= norm(vocab, axis=1, keepdims=True)

        return categories, vocab

    def train_classifier(self, train, test):
        dtype = self.rbms[0].dtype

        # --- find codes
        images, labels = train
        n_labels = len(np.unique(labels))
        codes = self.encode(images.astype(dtype))

        codes = theano.shared(codes.astype(dtype), name='codes')
        labels = tt.cast(theano.shared(labels.astype(dtype), name='labels'), 'int32')

        # --- compute backprop function
        Wshape = (self.rbms[-1].n_hid, n_labels)
        x = tt.matrix('x', dtype=dtype)
        y = tt.ivector('y')
        W = tt.matrix('W', dtype=dtype)
        b = tt.vector('b', dtype=dtype)

        W0 = np.random.normal(size=Wshape).astype(dtype).flatten() / 10
        b0 = np.zeros(n_labels)

        split_p = lambda p: [p[:-n_labels].reshape(Wshape), p[-n_labels:]]
        form_p = lambda params: np.hstack([p.flatten() for p in params])

        # compute negative log likelihood
        x_n = x + self.theano_rng.normal(size=x.shape, std=1, dtype=dtype)
        p_y_given_x = tt.nnet.softmax(tt.dot(x_n, W) + b)
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
        dtype = self.rbms[0].dtype
        params = []
        for rbm in self.rbms:
            params.extend([rbm.W, rbm.c])

        # --- compute backprop function
        assert self.W is not None and self.b is not None
        W = theano.shared(self.W.astype(dtype), name='Wc')
        b = theano.shared(self.b.astype(dtype), name='bc')

        x = tt.matrix('batch', dtype=dtype)
        y = tt.ivector('labels')

        # compute coding error
        code = self.propup(x)
        code_n = code + self.theano_rng.normal(size=code.shape, std=1, dtype=dtype)
        p_y_given_x = tt.nnet.softmax(tt.dot(code_n, W) + b)
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

        # --- find target codes
        images, labels = train
        labels = labels.astype('int32')
        batch_size = 50000

        import itertools
        ibatches = images.reshape(-1, batch_size, images.shape[1])
        lbatches = labels.reshape(-1, batch_size)
        ibatches = itertools.cycle(ibatches)
        lbatches = itertools.cycle(lbatches)

        def f_df_wrapper(p):
            for param, value in zip(params, split_p(p)):
                param.set_value(value.astype(param.dtype))

            batch = ibatches.next()
            label = lbatches.next()

            outs = f_df(batch, label)
            cost, grads = outs[0], outs[1:]
            grad = form_p(grads)
            return cost.astype('float64'), grad.astype('float64')

        p0 = form_p(np_params)
        p_opt, mincost, info = scipy.optimize.lbfgsb.fmin_l_bfgs_b(
            f_df_wrapper, p0, maxfun=100, iprint=1)

        for param, value in zip(params, split_p(p_opt)):
            param.set_value(value.astype(param.dtype), borrow=False)

    def test(self, train_set, test_set, classifier=False):
        images, labels = test_set
        codes = self.encode(images)

        if classifier:
            categories = np.unique(train_set[1])
            inds = np.argmax(np.dot(codes, self.W) + self.b, axis=1)
            return (labels != categories[inds])
        else:
            # find vocabulary pointers on training set
            categories, vocab = self.get_categories_vocab(train_set)

            # encode test set and compare to vocab pointers
            inds = np.argmax(np.dot(codes, vocab.T), axis=1)
            return (labels != categories[inds])

# --- load the data
filename = 'mnist.pkl.gz'

if not os.path.exists(filename):
    url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
    urllib.urlretrieve(url, filename=filename)

with gzip.open(filename, 'rb') as f:
    train, valid, test = pickle.load(f)

# --- pretrain with CD
shapes = [(28, 28), 500, 200, 50]
n_layers = len(shapes) - 1
rf_shapes = [(9, 9), None, None]
hidlinear = [False, False, True]
rates = [0.1, 0.1, 0.001]
assert len(rf_shapes) == n_layers
assert len(hidlinear) == n_layers
assert len(rates) == n_layers

train_images, train_labels = train
test_images, _ = test
test_batch = test_images[:200]

n_epochs = 15
batch_size = 100

dbn = DBN()
data = train_images
for i in range(n_layers):
    savename = "pretrained_rbm_%d.npz" % i
    if not os.path.exists(savename):
        batches = data.reshape(
            data.shape[0] / batch_size, batch_size, data.shape[1])

        rbm = RBM(shapes[i], shapes[i+1],
                  rf_shape=rf_shapes[i], hidlinear=hidlinear[i])
        dbn.rbms.append(rbm)
        rbm.pretrain(batches, dbn, test_batch,
                     n_epochs=n_epochs, rate=rates[i])
        rbm.save(savename)
    else:
        rbm = RBM.load(savename)
        dbn.rbms.append(rbm)

    data = rbm.encode(data)

plt.figure(99)
plt.clf()
recons = dbn.reconstruct(test_batch)
plotting.compare([test_batch.reshape(-1, 28, 28),
                  recons.reshape(-1, 28, 28)],
                 rows=5, cols=20)

print "mean error", dbn.test(train, test).mean()

# --- train classifier with backprop
savename = 'pretrained_classifier.npz'
if not os.path.exists(savename):
    dbn.train_classifier(train, test)
    np.savez(savename, W=dbn.W, b=dbn.b)
else:
    savedata = np.load(savename)
    dbn.W, dbn.b = savedata['W'], savedata['b']

print "mean error (classifier)", dbn.test(train, test, classifier=True).mean()

# --- train with backprop
if 1:
    dbn.backprop(train, test, n_epochs=100)

    print "mean error", dbn.test(train, test).mean()
    print "mean error (classifier)", dbn.test(train, test, classifier=True).mean()

    for i, rbm in enumerate(dbn.rbms):
        rbm.save('rbm_%d.npz' % i)
    np.savez('classifier.npz', W=dbn.W, b=dbn.b)

    # np.savez('dbn.npz',
    #          weights=[rbm.W.get_value() for rbm in dbn.rbms],
    #          biases=[rbm.c.get_value() for rbm in dbn.rbms],
    #          Wc=dbn.W,
    #          bc=dbn.b)

if 1:
    # compute top layers mean and std
    codes = dbn.encode(train_images)
    classes = np.dot(codes, dbn.W) + dbn.b

    plt.figure(108)
    plt.clf()
    plt.subplot(211)
    plt.hist(codes.flatten(), 100)
    plt.subplot(212)
    plt.hist(classes.flatten(), 100)

    print "code (mean, std):", codes.mean(), codes.std()
    print "class (mean, std):", classes.mean(), classes.std()
