from datetime import datetime
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utilities import pad_to_shape


def visualize_images(tb_summary, samples, denorm, dataset):
    images = []
    shape = (3, 513, 513)
    for (img, gt_label, pred) in samples:
        img = (denorm(img) * 255).astype(np.uint8)
        gt_label = dataset.decode_target(gt_label).transpose(2, 0, 1).astype(np.uint8)
        pred = dataset.decode_target(pred).transpose(2, 0, 1).astype(np.uint8)

        img = pad_to_shape(img, shape)
        gt_label = pad_to_shape(gt_label, shape)
        pred = pad_to_shape(pred, shape)

        concat_img = np.concatenate((img, pred, gt_label), axis=1)
        images.append(concat_img)
    images = np.stack(images, axis=0)
    tb_summary.add_images('Images, Predictions and Labels', images)


class TensorboardSummary(object):
    def __init__(self, path):
        self.path = path

    def create(self):
        curr_time = datetime.now().strftime("%d-%m-%Y_%H-%M")
        log_dir = os.path.join(self.path, curr_time)
        train_summary = SummaryWriter(os.path.join(log_dir, 'train'))
        val_summary = SummaryWriter(os.path.join(log_dir, 'val'))

        return train_summary, val_summary, log_dir


if __name__ == '__main__':
    tb = TensorboardSummary('logs')
    train_summary, val_summary, log_dir = tb.create()
    train_summary.close()
    val_summary.close()
