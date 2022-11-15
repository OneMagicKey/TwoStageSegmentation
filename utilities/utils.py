from torchvision.transforms.functional import normalize
import torch.nn as nn
import numpy as np
import os 

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt


def denormalize(tensor, mean, std):
    mean = np.array(mean)
    std = np.array(std)

    _mean = -mean/std
    _std = 1/std
    return normalize(tensor, _mean, _std)


class Denormalize(object):
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean/std
        self._std = 1/std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
        return normalize(tensor, self._mean, self._std)


def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum


def fix_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def pad_to_shape(image, shape):
    """Pad numpy array (image) to the given shape"""
    y_pad = shape[1] - image.shape[1]
    x_pad = shape[2] - image.shape[2]
    return np.pad(image, ((0, 0), (y_pad//2, y_pad//2 + y_pad%2), (x_pad//2, x_pad//2 + x_pad%2)))


def save_img(image, target, pred, loader, denorm, save_dir, img_id):
    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
    target = loader.dataset.decode_target(target).astype(np.uint8)
    pred = loader.dataset.decode_target(pred).astype(np.uint8)

    Image.fromarray(image).save(f'{save_dir}/{img_id}_image.png')
    Image.fromarray(target).save(f'{save_dir}/{img_id}_target.png')
    Image.fromarray(pred).save(f'{save_dir}/{img_id}_pred.png')

    fig = plt.figure()
    plt.imshow(image)
    plt.axis('off')
    plt.imshow(pred, alpha=0.7)
    ax = plt.gca()
    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
    plt.savefig(f'{save_dir}/{img_id}_overlay.png', bbox_inches='tight', pad_inches=0)
    plt.close()
