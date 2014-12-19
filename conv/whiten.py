"""
Algorithms for whitening.

ZCA combines a technique used in recent papers from Andrew Ng's group at
Stanford, with techniques developed by Nicolas Pinto at MIT.
"""
import warnings

import numpy as np

import matplotlib.pyplot as plt


def contrast_normalize(images, remove_mean=True, beta=10.0, hard_beta=True, show=False):
    X = images
    if X.ndim != 2:
        raise TypeError('contrast_normalize requires flat patches')

    if remove_mean:
        xm = X.mean(axis=1)
    else:
        xm = X[:,0] * 0

    Xc = X - xm[:, None]
    l2 = (Xc * Xc).sum(axis=1)

    if show:
        plt.figure(num=int('cn',36))
        plt.clf()
        plt.hist(l2)
        plt.title('l2 of image patches')

    if hard_beta:
        div2 = np.maximum(l2, beta)
    else:
        div2 = l2 + beta

    X = Xc / np.sqrt(div2[:, None])
    return X


def zca(images, gamma=1e-5, show=False, dtype='float64', **kwargs):
    # -- ZCA whitening (with band-pass)

    # Algorithm from Coates' sc_vq_demo.m
    shape_in = images.shape
    X = images.reshape((images.shape[0], -1)).astype(dtype)

    X = contrast_normalize(X, show=show, **kwargs)

    mu = X.mean(axis=0)
    X = X - mu[None, :]

    S = np.dot(X.T, X) / (X.shape[0] - 1)
    e, V = np.linalg.eigh(S)

    if show:
        plt.figure(num=int('zca',36))
        plt.clf()
        plt.semilogy(e[::-1])
        plt.title('Eigenspectrum of image patches')

    Sinv = np.dot(np.sqrt(1.0 / (e + gamma)) * V, V.T)
    X = np.dot(X, Sinv)

    if show:
        plt.figure(num=int('zcad',36))
        plt.clf()
        plt.hist(X.flatten())
        plt.title('Distribution of image pixels')

    return X.reshape(shape_in)


def test_cifar10():
    from skdata.cifar10.dataset import CIFAR10
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    data = CIFAR10()
    data.meta

    test_mask = np.array([m['split'] == 'test' for m in data.meta])
    train_images = data._pixels[~test_mask]
    train_labels = data._labels[~test_mask]

    # whiten
    images = train_images[:10000]
    images = images / 255.

    whites = zca(images, gamma=1e-5, show=True)

    def show_format(image):
        return (image - image.min()) / (image.max() - image.min())

    plt.figure(1)
    plt.clf()
    plt.subplot(211)
    # plt.imshow(images[0])
    plt.imshow(show_format(images[0]))
    plt.subplot(212)
    plt.imshow(show_format(whites[0]))
    plt.show()


if __name__ == '__main__':
    test_cifar10()
