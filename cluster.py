"""
Cluster analysis of MNIST dataset using the algorithm described in
   Alex Rodriguez and Alessandro Laio, "Clustering by fast search and find
       of density peaks", 2014, Science vol. 344 no. 6191 pp. 1492-6.
"""

import os
import gzip
import cPickle as pickle
import urllib

import numpy as np
import matplotlib.pyplot as plt
plt.ion()

# --- load the data
filename = 'mnist.pkl.gz'

if not os.path.exists(filename):
    url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
    urllib.urlretrieve(url, filename=filename)

with gzip.open(filename, 'rb') as f:
    train, valid, test = pickle.load(f)

# train_images, _ = train
images, labels = train
n = 1000
images = images[:n]
labels = labels[:n]
print images.dtype

# normalize
if 0:
    images /= np.sqrt(np.sum(images**2, axis=-1, keepdims=1))

# --- clustering
images -= images.mean(axis=-1, keepdims=1)
stds = images.std(axis=-1)

def distance_to_all(x):
    # return np.sqrt(np.sum((x - ys)**2, axis=-1))

    return np.dot(images, x) / images.shape[0] / stds / x.std()  # ncc

n = images.shape[0]

# distance matrix
d = np.zeros((n, n))
for i, image in enumerate(images):
    d[i] = distance_to_all(image)

if 1:
    plt.figure(101)
    plt.clf()
    plt.hist(d.flatten(), bins=100)

# rho of each point
# dc = 7.5
# dc = 0.8
dc = 0.1
rho = np.zeros(n)
for i, image in enumerate(images):
    # d = distance(image, images)
    rho[i] = (d[i] < dc).sum()

if 1:
    plt.figure(102)
    plt.clf()
    plt.hist(rho, bins=30)

# delta of each point
delta = np.zeros(n)
for i, image in enumerate(images):
    m = rho > rho[i]
    if m.any():
        delta[i] = d[i, m].min()
    else:
        delta[i] = d.max()


plt.figure(1)
plt.clf()
plt.plot(rho, delta, '.')

plt.figure(2)
plt.clf()
plt.scatter(rho, delta, c=labels)
plt.xlim(-1)
