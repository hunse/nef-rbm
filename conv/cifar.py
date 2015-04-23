"""Training a basic convolutional net on the CIFAR-10 dataset.

Notes:
- Weight cost is important, especially in early training to avoid instability.
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

# import whiten

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


def get_cifar10(white=True):
    if white:
        data = np.load('cifar10_white.npz')
    else:
        data = np.load('cifar10.npz')

    train = (data['train_x'], data['train_y'])
    valid = (data['valid_x'], data['valid_y'])
    test = (data['test_x'], data['test_y'])
    return train, valid, test


class Layer(object):
    # def __init__(self, epsW=0.001, epsB=0.002, momW=0.9, momB=0.9, initW=0.0001, initB=0.1, wc=0.0):
    def __init__(self, epsW=0.01, epsB=0.02, momW=0.9, momB=0.9, initW=0.0001, initB=0.1, wc=0.0):
        self.epsW = epsW
        self.epsB = epsB
        self.momW = momW
        self.momB = momB
        self.initW = initW
        self.initB = initB
        self.wc = wc

    def build(self, shape_in, rng=None):
        self.epsW = theano.shared(np.array(self.epsW, dtype=dtype))
        self.epsB = theano.shared(np.array(self.epsB, dtype=dtype))
        self.dweights = theano.shared(
            np.zeros_like(self.weights.get_value(borrow=True)))
        self.dbiases = theano.shared(
            np.zeros_like(self.biases.get_value(borrow=True)))

    def updates(self, grads):
        # updates[dp] = momentum * dp - (1 - momentum) * gp
        # updates[p] = p + self.alpha * updates[dp]
        # updates[dp] = momentum * dp - self.alpha * (decay * p + gp)
        # updates[p] = p + updates[dp]

        u = collections.OrderedDict()
        u[self.dweights] = self.momW * self.dweights - self.epsW * (
            self.wc * self.weights + grads[self.weights])
        u[self.weights] = self.weights + u[self.dweights]
        u[self.dbiases] = self.momB * self.dbiases - self.epsB * grads[self.biases]
        u[self.biases] = self.biases + u[self.dbiases]
        return u


class ConvLayer(Layer):
    def __init__(self, filters, filter_size=5, pooling=2, **kwargs):
        Layer.__init__(self, **kwargs)
        self.filters = filters
        self.filter_size = filter_size
        self.pooling = pooling

    def build(self, shape_in, rng=None):
        from theano import shared

        self.shape_in = shape_in
        self.channels = shape_in[0]

        self.weights = shared(
            rng.randn(*self.filter_shape).astype(dtype) * self.initW)
        self.biases = shared(np.ones(self.filters, dtype=dtype) * self.initB)

        Layer.build(self, shape_in, rng=rng)

    @property
    def filter_shape(self):
        return (self.filters, self.channels, self.filter_size, self.filter_size)

    @property
    def shape_out(self):
        pool_size = lambda x, p: int(np.ceil(x / float(p)))
        s1 = pool_size(self.shape_in[1] - (self.filter_size - 1), self.pooling)
        s2 = pool_size(self.shape_in[2] - (self.filter_size - 1), self.pooling)
        return (self.filters, s1, s2)

    @property
    def size_out(self):
        return np.prod(self.shape_out)

    def __call__(self, x, batchsize, noise=False):
        # from theano.tensor import lscalar, tanh, dot, grad, log, arange
        from theano.tensor.nnet.conv import conv2d
        # from hinge import multi_hinge_margin
        from pooling import max_pool_2d, average_pool_2d, power_pool_2d

        c = conv2d(x, self.weights,
                   image_shape=(batchsize,) + self.shape_in,
                   filter_shape=self.filter_shape)
        # t = tanh(c + self.biases[i].dimshuffle(0, 'x', 'x'))
        # y = tanh(max_pool_2d(t, (pooling[i], pooling[i])))

        t = relu(c + self.biases.dimshuffle(0, 'x', 'x'), noise=noise)
        # t = c + self.biases[i].dimshuffle(0, 'x', 'x')
        # y = relu(max_pool_2d(t, (pooling[i], pooling[i])))

        y = max_pool_2d(t, (self.pooling, self.pooling))
        # y = average_pool_2d(t, (pooling[i], pooling[i]))
        # y = power_pool_2d(t, (pooling[i], pooling[i]), p=2, b=0.001)

        return y


class FullLayer(Layer):
    def __init__(self, outputs, neuron=None, **kwargs):
        Layer.__init__(self, **kwargs)
        self.outputs = outputs
        self.neuron = neuron

    def build(self, shape_in, rng=None):
        from theano import shared

        shape = (np.prod(shape_in), self.outputs)
        self.weights = shared(rng.randn(*shape).astype(dtype) * self.initW)
        self.biases = shared(np.ones(self.outputs, dtype=dtype) * self.initB)

        Layer.build(self, shape_in, rng=rng)

    @property
    def shape_out(self):
        return (self.outputs,)

    @property
    def size_out(self):
        return np.prod(self.shape_out)

    def __call__(self, x, batchsize, noise=False):
        y = tt.dot(x.flatten(2), self.weights) + self.biases
        if self.neuron is not None:
            y = self.neuron(y, noise=noise)
        return y


class Convnet(object):
    def __init__(self, shape_in, layers, rng=None):
        from theano import shared
        if rng is None:
            rng = np.random.RandomState(9)

        self.shape_in = shape_in
        self.layers = layers

        for layer in layers:
            layer.build(shape_in, rng=rng)
            shape_in = layer.shape_out

    @property
    def params(self):
        return ([l.weights for l in self.layers] +
                [l.biases for l in self.layers])

    # def save(self, filename):
    #     d = dict(self.__dict__)
    #     d['weights'] = [o.get_value(borrow=True) for o in self.weights]
    #     d['biases'] = [o.get_value(borrow=True) for o in self.biases]
    #     # d['types'] = self.types
    #     # d['wc'] = self.wc.get_value(borrow=True)
    #     # d['bc'] = self.bc.get_value(borrow=True)
    #     np.savez(filename, **d)

    # @classmethod
    # def load(cls, filename):
    #     data = np.load(filename)
    #     self = cls.__new__(cls)
    #     self.__dict__.update(data)
    #     self.weights = [theano.shared(v) for v in self.weights]
    #     self.biases = [theano.shared(v) for v in self.biases]
    #     # self.wc = theano.shared(self.wc)
    #     # self.bc = theano.shared(self.bc)
    #     return self

    def get_train(self, batchsize, testsize):
        from theano.tensor import lscalar, tanh, dot, grad, log, arange
        from theano.tensor.nnet import softmax
        from theano.tensor.nnet.conv import conv2d
        from hinge import multi_hinge_margin
        from pooling import max_pool_2d, average_pool_2d, power_pool_2d

        sx = tt.tensor4()
        sy = tt.ivector()

        def propup(batchsize, noise=False, dropout=False):
            y = sx
            for layer in self.layers:
                y = layer(y, batchsize, noise=noise)
            return y

        yc = propup(batchsize, noise=True)
        if 1:
            cost = -log(softmax(yc))[arange(sy.shape[0]), sy].mean()
        else:
            cost = multi_hinge_margin(yc, sy).mean()
        error = tt.neq(tt.argmax(yc, axis=1), sy).mean()

        # get updates
        params = self.params
        grads = dict(zip(params, grad(cost, params)))
        updates = collections.OrderedDict()
        for layer in self.layers:
            updates.update(layer.updates(grads))

        train = theano.function(
            [sx, sy], [cost, error], updates=updates)

        # --- make test function
        y_pred = tt.argmax(propup(testsize, noise=False), axis=1)
        error = tt.mean(tt.neq(y_pred, sy))
        test = theano.function([sx, sy], error)

        return train, test


[train_x, train_y], [valid_x, valid_y], [test_x, test_y] = get_cifar10(white=True)
train_x, train_y = np.vstack((train_x, valid_x)), np.hstack((train_y, valid_y))
assert train_x.shape[2] == train_x.shape[3]

batch_size = 100
batch_x = train_x.reshape(-1, batch_size, *train_x.shape[1:])
batch_y = train_y.reshape(-1, batch_size)

test_size = 10000
test_batch_x = test_x.reshape(-1, test_size, *test_x.shape[1:])
test_batch_y = test_y.reshape(-1, test_size)

# net = Convnet(size, chan, filters=[10], pooling=[3])
# net = Convnet(size, chan, filters=[32, 32], pooling=[3, 3])
# net = Convnet(size, chan, filters=[32, 32], rfs=[5, 5], pooling=[3, 3], initW=[0.0001, 0.01])
# net = Convnet(size, chan, filters=[32], rfs=[5], pooling=[3], initW=[0.0001])

# net = Convnet(size, chan, filters=[32, 32], rfs=[9, 5], pooling=[3, 2], initW=[0.0001, 0.01])

# net = Convnet(size, chan, filters=[64, 64], rfs=[5, 5], pooling=[3, 3], initW=[0.0001, 0.01])

layers = [
    ConvLayer(filters=32, filter_size=9, pooling=3, initW=0.0001),
    ConvLayer(filters=32, filter_size=5, pooling=2, initW=0.01),
    # FullLayer(outputs=2048, initW=0.01, wc=0.004),
    FullLayer(outputs=10, initW=0.1, initB=0., wc=0.01),
    ]

# layers = [
#     ConvLayer(filters=64, filter_size=5, pooling=3, initW=0.0001),
#     ConvLayer(filters=64, filter_size=5, pooling=3, initW=0.01),
#     ]

net = Convnet(train_x.shape[1:], layers)
train, test = net.get_train(batch_size, test_size)

print "Starting..."
n_epochs = 50
test_errors = -np.ones(n_epochs)

for epoch in range(n_epochs):
    cost = 0.0
    error = 0.0
    for x, y in zip(batch_x, batch_y):
        costi, errori = train(x, y)
        cost += costi
        error += errori
    error /= batch_x.shape[0]

    test_errors[epoch] = test(test_batch_x[0], test_batch_y[0])
    eps = net.layers[0].epsW.get_value().item()
    print "Epoch %d (eps=%0.1e): %f, %f, %f" % (
        epoch, eps, cost, error, test_errors[epoch])

    if epoch > 0 and test_errors[epoch] >= test_errors[epoch-1]:
        for layer in net.layers:
            div = np.array(10., dtype=dtype)
            layer.epsW.set_value(layer.epsW.get_value() / div)
            layer.epsB.set_value(layer.epsB.get_value() / div)

        if any(layer.epsW.get_value() < 1e-8 for layer in net.layers):
            break

error = np.mean([test(x, y) for x, y in zip(test_batch_x, test_batch_y)])
print "Test error: %f" % error

# net.save('mnist-base.npz')

# net2 = Convnet.load('mnist-base.npz')
# _, test2 = net2.get_train(batch_size, test_size)
# error2 = np.mean([test2(x, y) for x, y in zip(test_batch_x, test_batch_y)])
# print "Test error: %f" % error2
