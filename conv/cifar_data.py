import warnings

import numpy as np
from skdata.cifar10.dataset import CIFAR10

import whiten


def write_cifar10(filename, dtype='float32', white=False):
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    data = CIFAR10()
    data.meta

    test_mask = np.array([m['split'] == 'test' for m in data.meta])
    train_x = data._pixels[~test_mask]
    train_y = data._labels[~test_mask]
    test_x = data._pixels[test_mask]
    test_y = data._labels[test_mask]

    # shuffle
    def shuffle(x, y, rng=np.random):
        i = rng.permutation(len(x))
        return x[i], y[i]

    rng = np.random.RandomState(98)
    train_x, train_y = shuffle(train_x, train_y, rng=rng)
    test_x, test_y = shuffle(test_x, test_y, rng=rng)

    # scale and whiten
    def process(images):
        images = images.astype(dtype) / 255.  # scale
        images = np.rollaxis(images, -1, 1)   # roll channel before (i,j)
        return images

    train_x, test_x = process(train_x), process(test_x)
    if white:
        train_x, test_x = whiten.zca(train_x, test_x, gamma=1e-4, dtype=dtype)

    # split out validation data from training
    m = np.zeros(len(train_x), dtype=bool)
    for label in np.unique(train_y):
        i, = (train_y == label).nonzero()
        m[i[:1000]] = 1

    valid_x, valid_y = train_x[m], train_y[m]
    train_x, train_y = train_x[~m], train_y[~m]

    # counts = np.zeros(10)
    # for i in train_y:
    #     counts[i] += 1
    # print counts

    np.savez(filename,
             train_x=train_x, train_y=train_y,
             valid_x=valid_x, valid_y=valid_y,
             test_x=test_x, test_y=test_y)

    # train, valid, test = (train_x, train_y), (valid_x, valid_y), (test_x, test_y)
    # np.savez(filename, train=train, valid=valid, test=test)


def show_cifar10():
    raise NotImplementedError("needs to be redone")

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
        show(train_x[i], ax=plt.subplot(3, 3, i+1))
    plt.figure()
    for i in range(9):
        show(test_images[i], ax=plt.subplot(3, 3, i+1))
    plt.show()
    assert False


if __name__ == '__main__':
    write_cifar10('cifar10.npz', white=False)
    write_cifar10('cifar10_white.npz', white=True)
