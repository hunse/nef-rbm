"""Training a basic convolutional net on the MNIST dataset.
"""

import os
import warnings

import numpy as np

os.environ['THEANO_FLAGS'] = 'device=gpu, floatX=float32'
import theano
import theano.tensor as tt

dtype = theano.config.floatX


def get_mnist():
    from skdata.mnist.dataset import MNIST

    data = MNIST()
    data.meta

    train_images, train_labels, test_images, test_labels = [
        data.arrays[k] for k in
        ['train_images', 'train_labels', 'test_images', 'test_labels']]

    def process(images):
        # add unitary channel dimension
        images = images[:, None, :, :]

        # pad images to 32 x 32
        images2 = np.zeros(images.shape[0:2] + (32, 32), dtype=dtype)
        for image, image2 in zip(images, images2):
            image2[0, 2:-2, 2:-2] = image

        # scale
        images2 /= 255.

        return images2

    train_images, test_images = process(train_images), process(test_images)
    return (train_images, train_labels), (test_images, test_labels)


class Convnet(object):
    def __init__(self, chan, filters=[6], pooling=[2], rng=np.random):
        from theano import shared

        outputs = 10
        rfs = [7, 5]
        n_layers = len(filters)
        assert len(pooling) == n_layers

        pool_size = lambda x, p: int(np.ceil(x / float(p)))
        sizes = [32]
        sizes.append(pool_size(sizes[-1] - (rfs[0] - 1), pooling[0]))

        w0 = shared(rng.randn(filters[0], chan, rfs[0], rfs[0]).astype(dtype) * np.sqrt(6. / 25))
        b0 = shared(np.zeros(filters[0], dtype=dtype))

        if n_layers >= 2:
            w1 = shared(rng.randn(filters[1], filters[0], rfs[1], rfs[1]).astype(dtype) * np.sqrt(6. / 25))
            b1 = shared(np.zeros(filters[1], dtype=dtype))
            sizes.append(pool_size(sizes[-1] - (rfs[1] - 1), pooling[1]))

        print sizes
        nwc = sizes[-1]**2 * filters[-1]
        print "nwc", nwc
        wc = shared(rng.normal(scale=0.1, size=(nwc, outputs)).astype(dtype))
        bc = shared(np.zeros(outputs, dtype=dtype))

        self.filters = filters
        self.pooling = pooling
        self.sizes = sizes
        self.rfs = rfs

        self.weights = [w0] + ([w1] if n_layers >= 2 else [])
        self.biases = [b0] + ([b1] if n_layers >= 2 else [])
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
        from theano.tensor.signal.downsample import max_pool_2d
        from hinge import multi_hinge_margin

        sx = tt.tensor4()
        sy = tt.ivector()

        filters = self.filters
        pooling = self.pooling
        sizes = self.sizes
        rfs = self.rfs

        n_layers = len(self.weights)
        assert len(filters) == n_layers and len(pooling) == n_layers

        def propup(size):
            # c0 = conv2d(sx, w0, image_shape=(size, chan, sizes[0], sizes[0]))
            c0 = conv2d(sx, self.weights[0],
                        image_shape=(size, chan, sizes[0], sizes[0]),
                        filter_shape=(filters[0], chan, rfs[0], rfs[0]))
            t0 = tanh(c0 + self.biases[0].dimshuffle(0, 'x', 'x'))
            s0 = tanh(max_pool_2d(t0, (pooling[0], pooling[0])))
            y = s0

            if n_layers >= 2:
                # c1 = conv2d(s0, w1, image_shape=(size, chan, sizes[1], sizes[1]))
                c1 = conv2d(s0, self.weights[1],
                            image_shape=(size, chan, sizes[1], sizes[1]),
                            filter_shape=(6, chan, rfs[1], rfs[1]))
                t1 = tanh(c1 + self.biases[1].dimshuffle(0, 'x', 'x'))
                s1 = tanh(max_pool_2d(t1, (pooling[1], pooling[1])))
                y = s1

            return dot(y.flatten(2), self.wc) + self.bc

        yc = propup(batchsize)
        if 1:
            cost = -log(softmax(yc))[arange(sy.shape[0]), sy].mean()
        else:
            cost = multi_hinge_margin(yc, sy).mean()
        error = tt.neq(tt.argmax(yc, axis=1), sy).mean()

        params = self.params
        gparams = grad(cost, params)

        train = theano.function(
            [sx, sy], [cost, error],
            updates=[(p, p - alpha * gp) for p, gp in zip(params, gparams)])

        # --- make test function
        y_pred = tt.argmax(propup(testsize), axis=1)
        error = tt.mean(tt.neq(y_pred, sy))
        test = theano.function([sx, sy], error)

        return train, test


[train_images, train_labels], [test_images, test_labels] = get_mnist()
chan = train_images.shape[1]

if 0:
    def show(image, ax=None):
        ax = plt.gca() if ax is None else ax
        if image.shape[0] == 1:
            ax.imshow(image[0], cmap='gray')
        else:
            ax.imshow(np.rollaxis(image, 0, image.ndim))

    import matplotlib.pyplot as plt
    plt.figure()
    show(train_images[1])
    plt.show()
    assert False

batch_size = 100
batches = train_images.reshape(-1, batch_size, *train_images.shape[1:])
batch_labels = train_labels.reshape(-1, batch_size)

test_size = 1000
test_batches = test_images.reshape(-1, test_size, *test_images.shape[1:])
test_batch_labels = test_labels.reshape(-1, test_size)

# --- mnist
net = Convnet(chan, filters=[10], pooling=[3])
# train, test = get_train(batch_size, test_size, chan, filters=[6, 16], pooling=[2, 2], alpha=0.01)

train, test = net.get_train(batch_size, test_size, alpha=5e-2)

n_epochs = 1
for epoch in range(n_epochs):
    cost = 0.0
    error = 0.0
    for x, y in zip(batches, batch_labels):
        costi, errori = train(x, y)
        cost += costi
        error += errori
    error /= batches.shape[0]

    test_error = test(test_batches[0], test_batch_labels[0])
    print "Epoch %d: %f, %f, %f" % (epoch, cost, error, test_error)

error = np.mean([test(x, y) for x, y in zip(test_batches, test_batch_labels)])
print "Test error: %f" % error

net.save('mnist-base.npz')

# net2 = Convnet.load('mnist-base.npz')
# _, test2 = net2.get_train(batch_size, test_size)
# error2 = np.mean([test2(x, y) for x, y in zip(test_batches, test_batch_labels)])
# print "Test error: %f" % error2
