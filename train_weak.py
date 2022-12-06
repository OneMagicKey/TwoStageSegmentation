from tqdm import tqdm
import network
import os
import random
import logging
import itertools
import argparse
import numpy as np

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes
from utilities import TensorboardSummary, visualize_images
from utilities import Denormalize, set_bn_momentum, save_img, PolyLR, FocalLoss
from utilities import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
from torch.nn.functional import softmax
import torchvision.transforms.functional as F


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--primary_model", type=str, default='deeplabv3plus_xception',
                        choices=available_models, help='primary model name')
    parser.add_argument("--ancillary_model", type=str, default='deeplabv3plus_xception_bbox',
                        choices=available_models, help='ancillary model name')
    parser.add_argument('--selfcorrection', type=str, default=None,
                        choices=['f', 'fw'], help='whether to use the self-correction module '
                                                  'to train the primary model')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=30000,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate (default: 0.001)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size_f", type=int, default=2,
                        help='batch size for the full set (default: 2)')
    parser.add_argument("--batch_size_w", type=int, default=3,
                        help='batch size for the weak set (default: 3)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: cross_entropy)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")
    parser.add_argument("--return_bbox", type=str, default=None, choices=['ground_truth', 'yolo'])
    parser.add_argument("--ckpt_ancillary", type=str, default=None,
                        help='[path to pretrained ancillary model')
    parser.add_argument("--ckpt_selfcorrecting", type=str, default=None,
                        help='[path to pretrained selfcorrection module')
    parser.add_argument("--multiscale_val", action='store_true', default=False,
                        help="enable multi-scale flipping inference during validation")
    parser.add_argument("--train_num_images", type=int, default=-1,
                        help="use fixed num of images for training (default: -1, i.e. use all available images")

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    # Visualization and logging options
    parser.add_argument("--enable_log", action='store_true', default=True,
                        help="use tensorboard for visualization and logging")
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    return parser


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'voc':
        train_transform = et.ExtCompose([
            et.ExtResize(size=(opts.crop_size, opts.crop_size)),
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                # et.ExtResizeImageOnly(512, max_size=513),
                et.ExtResizeImageOnly(size=(opts.crop_size, opts.crop_size)),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        train_dst_full = VOCSegmentation(root=opts.data_root, year=opts.year, image_set='train', download=opts.download,
                                         transform=train_transform, return_bbox=opts.return_bbox)
        train_dst_weak = VOCSegmentation(root=opts.data_root, year='2012_weak', image_set='train',
                                         download=opts.download,
                                         transform=train_transform, return_bbox=opts.return_bbox, )
        val_dst = VOCSegmentation(root=opts.data_root, year=opts.year, image_set='val', download=False,
                                  transform=val_transform, return_bbox=opts.return_bbox)

    return train_dst_full, val_dst, train_dst_weak


@torch.no_grad()
def validate(opts, model, loader, device, metrics, criterion, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        save_dir = 'results'
        os.makedirs(save_dir, exist_ok=True)
        denorm = Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_id = 0

    multiscale = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75] if opts.multiscale_val else [1.0]
    flipping = [0, 1] if opts.multiscale_val else [0]
    loss_val = 0.0
    for i, (images, labels, bboxes) in tqdm(enumerate(loader)):
        multi_avg = torch.zeros(images.shape[0], opts.num_classes, images.shape[2], images.shape[3],
                                device=device, dtype=torch.float32)  # BxCxHxW
        for (scale, flip) in itertools.product(multiscale, flipping):

            img_s = images.to(device, dtype=torch.float32)
            bboxes_s = bboxes.to(device, dtype=torch.float32)
            labels_s = labels.to(device, dtype=torch.long)

            img_s, labels_s, bboxes_s = et.ExtScale(scale=scale, is_tensor=True)(img_s, labels_s, bboxes_s)
            img_s, labels_s, bboxes_s = et.ExtRandomHorizontalFlip(p=flip)(img_s, labels_s, bboxes_s)

            outputs = model(img_s)

            loss_val += criterion(outputs, labels_s)  # just for the stats
            if flip:
                outputs = F.hflip(outputs)
            outputs = F.resize(outputs, size=multi_avg.shape[2:])
            multi_avg += outputs
        preds = multi_avg.argmax(dim=1)
        if not opts.crop_val:
            # it is required to resize prediction to initial size in order to get correct iou
            # batch size = 1 in this case
            images, preds, _ = et.ExtResize(size=labels.shape[1:])(images, preds, bboxes)
        images = images.cpu().numpy()
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy().astype(np.uint8)

        metrics.update(labels, preds)
        if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
            ret_samples.append(
                (images[0], labels[0], preds[0]))

        if opts.save_val_results:
            for image, target, pred in zip(images, labels, preds):
                save_img(image, target, pred, loader, denorm, save_dir, img_id)
                img_id += 1

    score = metrics.get_results()
    loss_val /= (len(multiscale) * len(flipping) * len(loader) / labels.shape[0])  # for normalization
    return score, ret_samples, loss_val


def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19

    # Setup logging to console
    logs = logging.getLogger()
    logs.setLevel(logging.INFO)
    stream = logging.StreamHandler()
    format = logging.Formatter("%(message)s")
    stream.setFormatter(format)
    logs.addHandler(stream)

    # Setup logging to file and tensorboard visualization
    if opts.enable_log:
        tb = TensorboardSummary('logs')
        tb_train, tb_val, log_dir = tb.create()

        log_file = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        log_file.setFormatter(format)
        logs.addHandler(log_file)

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logs.info(f'Device: {device}')

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    if opts.dataset == 'voc' and not opts.crop_val:
        opts.val_batch_size = 1

    train_dst_full, val_dst, train_dst_weak = get_dataset(opts)
    if opts.train_num_images > 0:
        train_dst_full, _ = data.random_split(train_dst_full,
                                              [opts.train_num_images, len(train_dst_full) - opts.train_num_images])
    sampler_full = data.RandomSampler(train_dst_full, replacement=True, num_samples=int(1e+10))
    sampler_weak = data.RandomSampler(train_dst_weak, replacement=True, num_samples=int(1e+10))
    train_loader_f = data.DataLoader(
        train_dst_full, batch_size=opts.batch_size_f, shuffle=False, num_workers=2,
        drop_last=True, sampler=sampler_full)
    train_loader_w = data.DataLoader(
        train_dst_weak, batch_size=opts.batch_size_w, shuffle=False, num_workers=2,
        drop_last=True, sampler=sampler_weak)
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
    logs.info(f'Dataset: {opts.dataset}, Train full set: {len(train_dst_full)}, '
              f'Train weak set: {len(train_dst_weak)}, Val set: {len(val_dst)}')

    # Set up model (all models are constructed at network.modeling)
    primary_model = network.modeling.__dict__[opts.primary_model](num_classes=opts.num_classes,
                                                                  output_stride=opts.output_stride)
    ancillary_model = network.modeling.__dict__[opts.ancillary_model](num_classes=opts.num_classes,
                                                                      output_stride=opts.output_stride)
    if opts.selfcorrection is not None:
        selfcorrection_module = network.modeling.__dict__['selfcorrection_module'](num_classes=opts.num_classes)

    if opts.separable_conv and 'plus' in opts.primary_model:
        network.convert_to_separable_conv(primary_model.classifier)
        network.convert_to_separable_conv(ancillary_model.classifier)
    set_bn_momentum(primary_model.backbone, momentum=0.01)
    set_bn_momentum(ancillary_model.backbone, momentum=0.01)

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': primary_model.backbone.parameters(), 'lr': opts.lr},
        {'params': primary_model.classifier.parameters(), 'lr': 10 * opts.lr}
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    if opts.selfcorrection is not None:
        optimizer.add_param_group({'params': selfcorrection_module.parameters(), 'lr': 10 * opts.lr})

    if opts.lr_policy == 'poly':
        scheduler = PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion,
    # criterion = utilities.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion_f = FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion_f = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
        criterion_w = nn.CrossEntropyLoss(reduction='mean')  # takes probs as target

    def save_ckpt(path, model):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        logs.info(f'Model saved as {path}')

    os.makedirs('checkpoints', exist_ok=True)
    # Restore
    best_score, cur_itrs, cur_epochs = 0.0, 0, 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        primary_model.load_state_dict(checkpoint["model_state"])
        primary_model = nn.DataParallel(primary_model)
        primary_model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            logs.info(f'Training state restored from {opts.ckpt}')
        logs.info(f'Model restored from {opts.ckpt}')
        del checkpoint  # free memory
    else:
        logs.info("[!] Retrain")
        primary_model = nn.DataParallel(primary_model)
        primary_model.to(device)
    checkpoint_ancillary = torch.load(opts.ckpt_ancillary, map_location=torch.device('cpu'))
    ancillary_model.load_state_dict(checkpoint_ancillary["model_state"])
    ancillary_model = nn.DataParallel(ancillary_model)
    ancillary_model.to(device)
    logs.info(f'Ancillary model restored from {opts.ckpt_ancillary}')
    if opts.ckpt_selfcorrecting and os.path.isfile(opts.ckpt_selfcorrecting):
        ckpt_selfcorrecting = torch.load(opts.ckpt_selfcorrecting, map_location=torch.device('cpu'))
        selfcorrection_module.load_state_dict(ckpt_selfcorrecting)
        logs.info(f'Selfcorrection module restored from {opts.ckpt_selfcorrecting}')
    if opts.selfcorrection is not None:
        selfcorrection_module.to(device)
    # ==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_log else None
    denorm = Denormalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.test_only:
        primary_model.eval()
        val_score, ret_samples, loss_val = validate(
            opts=opts, model=primary_model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=None)
        logs.info(metrics.to_str(val_score))
        return

    interval_loss = 0
    if opts.selfcorrection == 'f':
        loader = zip(train_loader_f)
        dst_len = len(train_dst_full)
        batch_size = opts.batch_size_f
    else:
        loader = zip(train_loader_f, train_loader_w)
        dst_len = len(train_dst_full) + len(train_dst_weak)
        batch_size = opts.batch_size_f + opts.batch_size_w

    ancillary_model.eval()
    cur_epochs += 1
    primary_model.train()
    if opts.selfcorrection is not None:
        selfcorrection_module.train()
    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        batch = next(loader)
        optimizer.zero_grad()
        if opts.selfcorrection == 'f':
            (images_f, labels_f, bboxes_f), = batch
            images_f = images_f.to(device, dtype=torch.float32)
            bboxes_f = bboxes_f.to(device, dtype=torch.float32)
            labels_f = labels_f.to(device, dtype=torch.long)

            with torch.no_grad():
                ancillary_logits_f = ancillary_model(images_f, bboxes_f)
                ancillary_logits_f = softmax(ancillary_logits_f, dim=1)

            outputs_f = primary_model(images_f)
            primary_logits_f = softmax(outputs_f, dim=1)

            outputs_sc_f = selfcorrection_module(primary_logits_f, ancillary_logits_f)

            loss_primary = criterion_f(outputs_f, labels_f)
            loss_sc = criterion_f(outputs_sc_f, labels_f)

            loss = loss_primary + loss_sc
            loss.backward()
            optimizer.step()

        elif opts.selfcorrection == 'fw':
            (images_f, labels_f, bboxes_f), (images_w, _, bboxes_w) = batch
            images_f = images_f.to(device, dtype=torch.float32)
            bboxes_f = bboxes_f.to(device, dtype=torch.float32)
            labels_f = labels_f.to(device, dtype=torch.long)
            images_w = images_w.to(device, dtype=torch.float32)
            bboxes_w = bboxes_w.to(device, dtype=torch.float32)
            with torch.no_grad():
                ancillary_logits_w = ancillary_model(images_w, bboxes_w)
                ancillary_logits_w = softmax(ancillary_logits_w, dim=1)

                ancillary_logits_f = ancillary_model(images_f, bboxes_f)
                ancillary_logits_f = softmax(ancillary_logits_f, dim=1)

            outputs_f = primary_model(images_f)
            outputs_w = primary_model(images_w)

            primary_logits_f = softmax(outputs_f, dim=1)

            with torch.no_grad():
                primary_logits_w = softmax(outputs_w, dim=1)
                labels_probs_w = selfcorrection_module(primary_logits_w, ancillary_logits_w)
                labels_probs_w = softmax(labels_probs_w, dim=1)

            outputs_sc_f = selfcorrection_module(primary_logits_f, ancillary_logits_f)

            loss_primary_f = criterion_f(outputs_f, labels_f)
            loss_primary_w = criterion_w(outputs_w, labels_probs_w)
            loss_sc = criterion_f(outputs_sc_f, labels_f)

            loss = loss_primary_f + loss_primary_w + loss_sc
            loss.backward()
            optimizer.step()

        else:
            (images_f, labels_f, _), (images_w, _, bboxes_w) = batch

            images_f = images_f.to(device, dtype=torch.float32)
            labels_f = labels_f.to(device, dtype=torch.long)
            images_w = images_w.to(device, dtype=torch.float32)
            bboxes_w = bboxes_w.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            with torch.no_grad():
                labels_probs_w = ancillary_model(images_w, bboxes_w)
                labels_probs_w = softmax(labels_probs_w, dim=1)

            outputs_f = primary_model(images_f)
            outputs_w = primary_model(images_w)

            loss_f = criterion_f(outputs_f, labels_f)
            loss_w = criterion_w(outputs_w, labels_probs_w)
            loss = loss_f + loss_w

            loss.backward()
            optimizer.step()
        cur_itrs += 1

        np_loss = loss.detach().cpu().numpy()
        interval_loss += np_loss

        if cur_itrs % 10 == 0:
            interval_loss = interval_loss / 10
            logs.info(f'Epoch {cur_epochs}, Itrs {cur_itrs}/{opts.total_itrs}, Loss={interval_loss:.5f}')
            if opts.enable_log:
                tb_train.add_scalar('Loss', interval_loss, cur_itrs)
            interval_loss = 0.0

        if cur_itrs % opts.val_interval == 0:
            save_ckpt(f'checkpoints/latest_{opts.primary_model}_{opts.dataset}_num_{len(train_dst_full)}.pth',
                      primary_model)
            if opts.selfcorrection is not None:
                path = f'checkpoints/latest_sc_model_{opts.selfcorrection}_{opts.dataset}_num_' \
                       f'{len(train_dst_full)}.pth'
                torch.save(selfcorrection_module.state_dict(), path)
                logs.info(f'Model saved as {path}')
            logs.info("validation...")
            primary_model.eval()
            val_score, ret_samples, loss_val = validate(
                opts=opts, model=primary_model, loader=val_loader, device=device, metrics=metrics,
                criterion=criterion_f,
                ret_samples_ids=vis_sample_id)
            logs.info(metrics.to_str(val_score))
            if val_score['Mean IoU'] > best_score:  # save best model
                best_score = val_score['Mean IoU']
                save_ckpt(f'checkpoints/best_{opts.primary_model}_{opts.dataset}_num_{len(train_dst_full)}.pth',
                          primary_model)
                if opts.selfcorrection is not None:
                    path = f'checkpoints/best_sc_model_{opts.selfcorrection}_{opts.dataset}_num_' \
                       f'{len(train_dst_full)}.pth'
                    torch.save(selfcorrection_module.state_dict(), path)
                    logs.info(f'Model saved as {path}')
            if opts.enable_log:  # visualize validation score and samples
                tb_val.add_scalar('Loss', loss_val, cur_itrs)
                tb_val.add_scalar('[Val] Overall Acc', val_score['Overall Acc'], cur_itrs)
                tb_val.add_scalar('[Val] Mean IoU', val_score['Mean IoU'], cur_itrs)

                visualize_images(tb_val, ret_samples, denorm, val_dst)
            primary_model.train()
        scheduler.step()

        if cur_epochs * dst_len // batch_size < cur_itrs:
            cur_epochs += 1
        if cur_itrs >= opts.total_itrs:
            tb_train.close()
            tb_val.close()
            return


if __name__ == '__main__':
    main()
