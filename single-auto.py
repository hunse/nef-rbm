"""
Try making an autoencoder using the NEF

Notes:
- setting intercepts to -0.5 improves training RMSE versus random or 0.0
- setting max rates higher improves training RMSE
  (why? training is just done on rates, so why should this make a difference)
- giving encoders limited receptive fields improves training RMSE
"""

import collections
import os
import gzip
import cPickle as pickle
import urllib

import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import plotting

import nengo
from nengo.utils.distributions import UniformHypersphere

# --- load the data
filename = 'mnist.pkl.gz'

if not os.path.exists(filename):
    url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
    urllib.urlretrieve(url, filename=filename)

with gzip.open(filename, 'rb') as f:
    train, valid, test = pickle.load(f)

images, _ = train
n_vis = images.shape[1]
n_hid = 500

if 1:
    # images -= images.mean(0)
    # images /= images.std(0) + 1e-3
    images = 2 * images - 1

images = images[:1000]

if 1:
    # generate
    rng = np.random

    rf_shape = (9, 9)
    M, N = 28, 28
    m, n = rf_shape

    # find random positions for top-left corner of each RF
    i = rng.randint(low=0, high=M-m+1, size=n_hid)
    j = rng.randint(low=0, high=N-n+1, size=n_hid)

    mask = np.zeros((n_hid, M, N), dtype='bool')
    for k in xrange(n_hid):
        mask[k, i[k]:i[k]+m, j[k]:j[k]+n] = True

    mask = mask.reshape(n_hid, n_vis)
    encoders = rng.normal(size=(n_hid, n_vis)) * mask

else:
    # rng = np.random
    # encoders = rng.normal(size=(n_hid, n_vis))
    # encoders = UniformHypersphere(n_vis, surface=True)
    encoders = None

# --- make the network

model = nengo.Network()

with model:
    u = nengo.Node(output=images[0])
    up = nengo.Probe(u)

    a = nengo.Ensemble(n_hid, n_vis, eval_points=images, encoders=encoders,
                       intercepts=[-0.5]*n_hid, max_rates=[200]*n_hid)

    o = nengo.Node(size_in=n_vis)
    op = nengo.Probe(o, synapse=0.03)

    ca = nengo.Connection(u, a)
    co = nengo.Connection(a, o)

from hunse_tools.timing import tic, toc
tic()
sim = nengo.Simulator(model)
toc()
tic()
sim.run(1.0)
toc()

x = sim.data[up].reshape(-1, 28, 28)
y = sim.data[op].reshape(-1, 28, 28)

plt.figure(1)
plt.clf()
plt.subplot(121)
plt.imshow(x.mean(0), cmap='gray')
plt.subplot(122)
plt.imshow(y.mean(0), cmap='gray')

# --- check RMSEs over a number of builds
n_trials = 10

rmses = np.zeros((n_trials, n_vis))

for i in range(n_trials):
    if a.encoders is not None:
        a.encoders = rng.normal(size=a.encoders.shape) * mask

    sim = nengo.Simulator(model)
    info = sim.data[co].solver_info
    rmses[i] = info['rmses']

mean = rmses.mean(1)
print "RMSEs: mean %f +/- %f" % (mean.mean(), 2 * mean.std() / np.sqrt(n_trials))
# print "RMSEs: mean %f, std %f, min %f, max %f" % (rmses.mean(), rmses.std(), rmses.min(), rmses.max())
