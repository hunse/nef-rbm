import os
import gzip
import cPickle as pickle
import urllib

import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import nengo

# --- parameters
presentation_time = 0.1
# Ncode = 10
Ncode = 30
# pstc = 0.006
pstc = 0.004

# --- functions
# def norm(x, **kwargs):
#     return np.sqrt((x**2).sum(**kwargs))

# def sigmoid(x):
#     return 1. / (1 + np.exp(-x))

# def forward(x, weights, biases):
#     for w, b in zip(weights, biases):
#         x = np.dot(x, w)
#         x += b
#         if w is not weights[-1]:
#             x = sigmoid(x)

#     return x

def get_image(t):
    return test_images[int(t / presentation_time)]

def test_dots(t, dots):
    i = int(t / presentation_time)
    j = np.argmax(dots)
    return test_labels[i] == labels[j]

# --- load the RBM data
data = np.load('nlif-deep.npz')
weights = data['weights']
biases = data['biases']
Wc = data['Wc']
bc = data['bc']

# --- load the testing data
filename = 'mnist.pkl.gz'

if not os.path.exists(filename):
    url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
    urllib.urlretrieve(url, filename=filename)

with gzip.open(filename, 'rb') as f:
    train, valid, test = pickle.load(f)

test_images, test_labels = test

# shuffle
rng = np.random.RandomState(92)
inds = rng.permutation(len(test_images))
test_images = test_images[inds]
test_labels = test_labels[inds]

labels = np.unique(test_labels)
n_labels = labels.size

# --- create the model
neuron_type = nengo.LIF(tau_rc=0.02, tau_ref=0.002)
# alpha = 1
# beta = 1
max_rate = 63
intercept = 0
assert np.allclose(neuron_type.gain_bias(max_rate, intercept), (1, 1), atol=1e-2)
amp = 1. / 65
neuron_params = dict(max_rates=max_rate, intercepts=intercept, neuron_type=neuron_type)

model = nengo.Network()
with model:
    input_images = nengo.Node(output=get_image, label='images')

    # W, b = weights[0], biases[0]
    # n = b.size
    # layer0 = nengo.Ensemble(n, 1, label='layer 0', neuron_type=neuron_type,
    #                         max_rates=max_rate*np.ones(n),
    #                         intercepts=intercept*np.ones(n))
    # bias = nengo.Node(output=b, label='bias 0')
    # nengo.Connection(bias, layer0.neurons, transform=np.eye(n), synapse=0)
    # nengo.Connection(input_images, layer0.neurons,
    #                  transform=W.T, synapse=pstc)

    # layers = [layer0]

    # --- make sigmoidal layers
    layers = []
    for i, [W, b] in enumerate(zip(weights[:-1], biases[:-1])):
        n = b.size
        layer = nengo.Ensemble(n, 1, label='layer %d' % i, neuron_type=neuron_type,
                               max_rates=max_rate*np.ones(n),
                               intercepts=intercept*np.ones(n))
        bias = nengo.Node(output=b)
        nengo.Connection(bias, layer.neurons, transform=np.eye(n), synapse=0)

        if i == 0:
            nengo.Connection(input_images, layer.neurons,
                             transform=W.T, synapse=pstc)
        else:
            nengo.Connection(layers[-1].neurons, layer.neurons,
                             transform=W.T * amp, synapse=pstc)

        layers.append(layer)

    # --- make code layer
    W, b = weights[-1], biases[-1]
    code_layer = nengo.networks.EnsembleArray(50, b.size, label='code', radius=15)
    code_bias = nengo.Node(output=b)
    nengo.Connection(code_bias, code_layer.input, synapse=0)
    nengo.Connection(layers[-1].neurons, code_layer.input,
                     transform=W.T * amp * 1000, synapse=pstc)

    # --- make cleanup
    class_layer = nengo.networks.EnsembleArray(100, 10, label='class', radius=15)
    class_bias = nengo.Node(output=bc)
    nengo.Connection(class_bias, class_layer.input, synapse=0)
    nengo.Connection(code_layer.output, class_layer.input,
                     transform=Wc.T, synapse=pstc)

    test = nengo.Node(output=test_dots, size_in=n_labels)
    nengo.Connection(class_layer.output, test)

    probe_code = nengo.Probe(code_layer.output, synapse=0.03)
    probe_class = nengo.Probe(class_layer.output, synapse=0.03)
    probe_test = nengo.Probe(test, synapse=0.01)


# --- simulation
# rundata_file = 'rundata.npz'
# if not os.path.exists(rundata_file):
if 1:
    sim = nengo.Simulator(model)
    # sim.run(1.)
    sim.run(5.)

    t = sim.trange()
    x = sim.data[probe_code]
    y = sim.data[probe_class]
    z = sim.data[probe_test]

    # np.savez(rundata_file, t=t, y=y, z=z)
else:
    rundata = np.load(rundata_file)
    t, y, z = [rundata[k] for k in ['t', 'y', 'z']]

# --- plots
def plot_bars():
    ylim = plt.ylim()
    for x in np.arange(0, t[-1], presentation_time):
        plt.plot([x, x], ylim, 'k--')

inds = slice(0, int(t[-1]/presentation_time) + 1)
images = test_images[inds]
labels = test_labels[inds]
allimage = np.zeros((28, 28 * len(images)), dtype=images.dtype)
for i, image in enumerate(images):
    allimage[:, i * 28:(i + 1) * 28] = image.reshape(28, 28)

z2 = np.argmax(y, axis=1) == labels.repeat(100)

plt.figure(1)
plt.clf()
r, c = 4, 1

plt.subplot(r, c, 1)
plt.imshow(allimage, aspect='auto', interpolation='none', cmap='gray')
plt.xticks([])
plt.yticks([])

plt.subplot(r, c, 2)
plt.plot(t, x)
plot_bars()
plt.ylabel('code')

plt.subplot(r, c, 3)
plt.plot(t, y)
plot_bars()
plt.ylabel('class')

plt.subplot(r, c, 4)
plt.plot(t, z)
plt.ylim([-0.1, 1.1])
plot_bars()
plt.xlabel('time [s]')
plt.ylabel('correct')

# plt.savefig('runtime.png')

# --- compute error rate
zblocks = z.reshape(-1, 100)[:, 50:]  # 50 ms blocks at end of each 100
errors = np.mean(zblocks, axis=1) < 0.5
print errors.mean()

zblocks = z2.reshape(-1, 100)[:, 80:]  # 20 ms blocks at end of each 100
errors = np.mean(zblocks, axis=1) < 0.5
print errors.mean()
