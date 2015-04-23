"""Training a basic convolutional net on the CIFAR-10 dataset.
"""
import collections
import os
import warnings

import numpy as np

# os.environ['THEANO_FLAGS'] = 'device=gpu, floatX=float32'
os.environ['THEANO_FLAGS'] = 'device=gpu, floatX=float32, exception_verbosity=high'
import theano
import theano.tensor as tt
import theano.sandbox.rng_mrg
dtype = theano.config.floatX

import whiten

t_rng = theano.sandbox.rng_mrg.MRG_RandomStreams()

def relu(x, noise=False):
    # return tt.maximum(x, 0)
    return tt.minimum(tt.maximum(x, 0), 6)


# def relu(x, noise=False):
#     # return tt.maximum(x, 0)
#     # return tt.minimum(tt.maximum(x, 0), 6)

#     if noise:
#         x = x + t_rng.normal(size=x.shape, std=0.1)

#     # y = tt.minimum(tt.maximum(x, 0), 6)
#     y = tt.minimum(tt.maximum(x, 0), 1)
#     # std = tt.and_(tt.gt(y, 0), tt.lt(y, 6))
#     # y +=

#     return y


def get_cifar10():
    from skdata.cifar10.dataset import CIFAR10
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    data = CIFAR10()
    data.meta

    test_mask = np.array([m['split'] == 'test' for m in data.meta])
    train_images = data._pixels[~test_mask]
    train_labels = data._labels[~test_mask]
    test_images = data._pixels[test_mask]
    test_labels = data._labels[test_mask]

    def process(images):
        images = images.astype(dtype) / 255.  # scale
        images = np.rollaxis(images, -1, 1)   # roll channel before (i,j)
        return images

    train_images, test_images = process(train_images), process(test_images)

    train_images, test_images = whiten.zca(
        train_images, test_images, gamma=1e-4, dtype=dtype)

    return (train_images, train_labels), (test_images, test_labels)


class Convnet(object):
    def __init__(self, size, chan,
                 filters=[6], rfs=[5], pooling=[2], initW=[0.0001],
                 rng=np.random):
        from theano import shared

        outputs = 10
        n_layers = len(filters)
        assert len(pooling) == n_layers

        pool_size = lambda x, p: int(np.ceil(x / float(p)))
        sizes = [size]
        chs = [chan] + filters
        weights = []
        d_weights = {}  # last change in weight (for momentum)
        biases = []
        for i in range(n_layers):
            shape = (chs[i+1], chs[i], rfs[i], rfs[i])
            sizes.append(pool_size(sizes[i] - (rfs[i] - 1), pooling[i]))

            weights.append(shared(rng.randn(*shape).astype(dtype) * initW[i]))
            d_weights[weights[i]] = shared(np.zeros(shape, dtype=dtype))

            # init biases to 1 to speed initial learning (Krizhevsky 2012)
            biases.append(shared(np.zeros(chs[i+1], dtype=dtype)))
            # biases.append(shared(np.ones(chs[i+1], dtype=dtype)))

        nwc = sizes[-1]**2 * filters[-1]
        wc = shared(rng.normal(scale=0.1, size=(nwc, outputs)).astype(dtype))
        bc = shared(np.zeros(outputs, dtype=dtype))

        self.alpha = shared(np.array(0.01, dtype=dtype))

        self.channels_in = chan
        self.filters = filters
        self.pooling = pooling
        self.sizes = sizes
        self.rfs = rfs

        self.weights = weights
        self.d_weights = d_weights
        self.biases = biases
        self.wc = wc
        self.bc = bc

    @property
    def params(self):
        return self.weights + self.biases + [self.wc, self.bc]

    def save(self, filename):
        d = dict(self.__dict__)
        d['weights'] = [o.get_value(borrow=True) for o in self.weights]
        d['biases'] = [o.get_value(borrow=True) for o in self.biases]
        d['wc'] = self.wc.get_value(borrow=True)
        d['bc'] = self.bc.get_value(borrow=True)
        np.savez(filename, **d)

    @classmethod
    def load(cls, filename):
        data = np.load(filename)
        self = cls.__new__(cls)
        self.__dict__.update(data)
        self.weights = [theano.shared(v) for v in self.weights]
        self.biases = [theano.shared(v) for v in self.biases]
        self.wc = theano.shared(self.wc)
        self.bc = theano.shared(self.bc)
        return self

    def get_train(self, batchsize, testsize, alpha=5e-2):
        from theano.tensor import lscalar, tanh, dot, grad, log, arange
        from theano.tensor.nnet import softmax
        from theano.tensor.nnet.conv import conv2d
        from hinge import multi_hinge_margin
        from pooling import max_pool_2d, average_pool_2d, power_pool_2d

        self.alpha.set_value(alpha)

        sx = tt.tensor4()
        sy = tt.ivector()

        pooling = self.pooling
        sizes = self.sizes
        rfs = self.rfs
        chs = [self.channels_in] + self.filters
        n_layers = len(self.weights)

        def propup(batchsize, noise=False):
            y = sx
            for i in range(n_layers):
                c = conv2d(y, self.weights[i],
                           image_shape=(batchsize, chs[i], sizes[i], sizes[i]),
                           filter_shape=(chs[i+1], chs[i], rfs[i], rfs[i]))
                # t = tanh(c + self.biases[i].dimshuffle(0, 'x', 'x'))
                # y = tanh(max_pool_2d(t, (pooling[i], pooling[i])))

                t = relu(c + self.biases[i].dimshuffle(0, 'x', 'x'), noise=noise)
                # t = c + self.biases[i].dimshuffle(0, 'x', 'x')
                # y = relu(max_pool_2d(t, (pooling[i], pooling[i])))

                y = max_pool_2d(t, (pooling[i], pooling[i]))
                # y = average_pool_2d(t, (pooling[i], pooling[i]))
                # y = power_pool_2d(t, (pooling[i], pooling[i]), p=2, b=0.001)

            return dot(y.flatten(2), self.wc) + self.bc

        yc = propup(batchsize, noise=True)
        if 1:
            cost = -log(softmax(yc))[arange(sy.shape[0]), sy].mean()
        else:
            cost = multi_hinge_margin(yc, sy).mean()
        error = tt.neq(tt.argmax(yc, axis=1), sy).mean()

        # get updates
        params = self.params
        gparams = grad(cost, params)
        updates = collections.OrderedDict()

        momentum = 0.9
        decay = 0.0005
        for p, gp in zip(params, gparams):

            if p in self.weights:
                dp = self.d_weights[p]
                updates[dp] = momentum * dp - (1 - momentum) * gp
                updates[p] = p + self.alpha * updates[dp]
                # updates[dp] = momentum * dp - self.alpha * (decay * p + gp)
                # updates[p] = p + updates[dp]
            else:
                updates[p] = p - self.alpha * gp

        train = theano.function(
            [sx, sy], [cost, error], updates=updates)

        # --- make test function
        y_pred = tt.argmax(propup(testsize, noise=False), axis=1)
        error = tt.mean(tt.neq(y_pred, sy))
        test = theano.function([sx, sy], error)

        return train, test


[train_images, train_labels], [test_images, test_labels] = get_cifar10()
chan = train_images.shape[1]
size = train_images.shape[2]
assert size == train_images.shape[3]

if 0:
    def show_format(image):
        # return (image - image.min()) / (image.max() - image.min())
        return ((image - image.mean()) / image.std() / 3 + 0.5).clip(0, 1)

    def show(image, ax=None):
        image = show_format(image)
        ax = plt.gca() if ax is None else ax
        if image.shape[0] == 1:
            ax.imshow(image[0], cmap='gray')
        else:
            ax.imshow(np.rollaxis(image, 0, image.ndim))

    import matplotlib.pyplot as plt
    plt.figure()
    for i in range(9):
        show(train_images[i], ax=plt.subplot(3, 3, i+1))
    plt.figure()
    for i in range(9):
        show(test_images[i], ax=plt.subplot(3, 3, i+1))
    plt.show()
    assert False

batch_size = 100
batches = train_images.reshape(-1, batch_size, *train_images.shape[1:])
batch_labels = train_labels.reshape(-1, batch_size)

test_size = 1000
test_batches = test_images.reshape(-1, test_size, *test_images.shape[1:])
test_batch_labels = test_labels.reshape(-1, test_size)

# net = Convnet(size, chan, filters=[10], pooling=[3])
# net = Convnet(size, chan, filters=[32, 32], pooling=[3, 3])
# net = Convnet(size, chan, filters=[32, 32], rfs=[5, 5], pooling=[3, 3], initW=[0.0001, 0.01])
# net = Convnet(size, chan, filters=[32], rfs=[5], pooling=[3], initW=[0.0001])

net = Convnet(size, chan, filters=[32, 32], rfs=[9, 5], pooling=[3, 2], initW=[0.0001, 0.01])


train, test = net.get_train(batch_size, test_size, alpha=5e-2)
# train, test = net.get_train(batch_size, test_size, alpha=5e-3)

print "Starting..."
n_epochs = 50
# train_errors
test_errors = -np.ones(n_epochs)

for epoch in range(n_epochs):
    cost = 0.0
    error = 0.0
    for x, y in zip(batches, batch_labels):
        costi, errori = train(x, y)
        cost += costi
        error += errori
    error /= batches.shape[0]

    test_errors[epoch] = test(test_batches[0], test_batch_labels[0])
    print "Epoch %d (alpha=%0.1e): %f, %f, %f" % (
        epoch, net.alpha.get_value(), cost, error, test_errors[epoch])

    if epoch > 0 and test_errors[epoch] >= test_errors[epoch-1]:
        net.alpha.set_value(net.alpha.get_value() / np.array(10., dtype=dtype))

error = np.mean([test(x, y) for x, y in zip(test_batches, test_batch_labels)])
print "Test error: %f" % error

# net.save('mnist-base.npz')

# net2 = Convnet.load('mnist-base.npz')
# _, test2 = net2.get_train(batch_size, test_size)
# error2 = np.mean([test2(x, y) for x, y in zip(test_batches, test_batch_labels)])
# print "Test error: %f" % error2
