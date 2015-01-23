"""
Algorithms for whitening.

ZCA combines a technique used in recent papers from Andrew Ng's group at
Stanford, with techniques developed by Nicolas Pinto at MIT.
"""
import warnings

import numpy as np

import matplotlib.pyplot as plt


def contrast_normalize(images, remove_mean=True, beta=10.0, hard_beta=True, show=False):
    X = np.array(images)
    if X.ndim != 2:
        raise TypeError('contrast_normalize requires flat patches')

    if remove_mean:
        X -= X.mean(axis=1)[:, None]

    l2 = (X * X).sum(axis=1)

    if show:
        plt.figure(num=int('cn',36))
        plt.clf()
        plt.hist(l2)
        plt.title('l2 of image patches')

    if hard_beta:
        div2 = np.maximum(l2, beta)
    else:
        div2 = l2 + beta

    X /= np.sqrt(div2[:, None])
    return X


def zca(train, test=None, gamma=1e-5, show=False, dtype='float64', **kwargs):
    # -- ZCA whitening (with band-pass)

    # Algorithm from Coates' sc_vq_demo.m
    X = train.reshape((train.shape[0], -1)).astype(dtype)

    X = contrast_normalize(X, show=show, **kwargs)

    mu = X.mean(axis=0)
    X -= mu[None, :]

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

    X = X.reshape(train.shape)

    if test is None:
        return X
    else:
        assert train.shape[1:] == test.shape[1:]
        Y = test.reshape((test.shape[0], -1)).astype(dtype)
        Y = contrast_normalize(Y, show=show, **kwargs)
        Y -= mu[None, :]
        Y = np.dot(Y, Sinv)
        Y = Y.reshape(test.shape)
        return X, Y


def test_cifar10():
    from skdata.cifar10.dataset import CIFAR10
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    data = CIFAR10()
    data.meta

    test_mask = np.array([m['split'] == 'test' for m in data.meta])
    train_images = data._pixels[~test_mask][:10000] / 255.
    test_images = data._pixels[test_mask][:1000] / 255.

    # whiten
    train_whites, test_whites = zca(
        train_images, test_images, gamma=1e-4, show=False)

    def show_format(image):
        # return (image - image.min()) / (image.max() - image.min())
        return ((image - image.mean()) / image.std() / 3 + 0.5).clip(0, 1)

    plt.figure(1)
    plt.clf()
    n = 5
    for i in range(n):
        plt.subplot(4, n, i+1)
        plt.imshow(show_format(train_images[i]))
        plt.subplot(4, n, i+n+1)
        plt.imshow(show_format(train_whites[i]))
        plt.subplot(4, n, i+2*n+1)
        plt.imshow(show_format(test_images[i]))
        plt.subplot(4, n, i+3*n+1)
        plt.imshow(show_format(test_whites[i]))
    plt.show()


if __name__ == '__main__':
    test_cifar10()
