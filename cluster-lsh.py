"""
Playing with the clustering method proposed in

    Alex Rodriguez, Alessandro Laio. Clustering by fast search and find of
        density peaks. Science, June 2014, vol. 344 no. 6191 pp. 1492-1496.
"""
import itertools

import numpy as np
import matplotlib.pyplot as plt

from nengo.utils.distributions import UniformHypersphere

from hunse_tools.timing import tic, toc

N = 3000
# N = 10000
# D = 128
# D = 64
D = 16

# dc = 1.2  # rho counting distance threshold
# dc = 0.8  # rho counting distance threshold
dc = 0.8  # rho counting distance threshold

# generate random points on unit hypersphere
points = UniformHypersphere(surface=True).sample(N, D)

# compute rho exactly
def distance(x, ys):
    return np.sqrt(np.sum((x - ys)**2, axis=-1))

tic()
rho = np.zeros(N)
for i, point in enumerate(points):
    d = distance(point, points)
    rho[i] = (d < dc).sum()
toc()

if 0:
    d = distance(points[0], points)

    plt.figure(98)
    plt.clf()
    plt.hist(d, bins=30)

# compute rho using random hyperplane LSH
D_lsh = 5
planes = UniformHypersphere(surface=True).sample(D_lsh, D)
powers = 2**np.arange(0, D_lsh)
ipowers = 2.0**np.arange(-D_lsh, 0)

BINARY_BINS = False

tic()
if BINARY_BINS:
    hashcodes = np.dot(points, planes.T) > 0
else:
    hashcodes = np.dot(points, planes.T)

    if 0:
        plt.figure(101)
        plt.clf()
        r = D_lsh
        for i in xrange(r):
            plt.subplot(r, 1, i+1)
            plt.hist(hashcodes[:,i], bins=30)

    hashcodes[np.abs(hashcodes) < 0.1] = 0
    hashcodes = np.sign(hashcodes)

unique_codes = np.unique(map(tuple, hashcodes))
rho1 = rho
rho2 = np.zeros(N)
for code in unique_codes:

    group_inds, = (hashcodes == code).all(axis=1).nonzero()

    if BINARY_BINS:
        m = np.abs(hashcodes - code).sum(axis=1) <= 2
    else:
        m = (np.abs(hashcodes - code) <= 1).all(axis=1)

    neighbourhood = points[m]
    print len(neighbourhood)
    for i, point in enumerate(points[group_inds]):
        d = distance(point, neighbourhood)
        rho2[group_inds[i]] = (d < dc).sum()

toc()

# print (rho1 != rho2).sum()
print np.abs(rho1 - rho2).sum()
print zip(rho1[:30], rho2)

plt.ion()
plt.figure(1)
plt.clf()
plt.subplot(211)
plt.hist(rho1, bins=30)
plt.subplot(212)
plt.hist(rho2, bins=30)

# i = np.argsort(
