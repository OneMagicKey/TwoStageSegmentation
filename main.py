from tqdm import tqdm
import network
import utilities
import os
import random
from datetime import datetime
import itertools
import argparse
import numpy as np

from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from datasets import VOCSegmentation, Cityscapes
from utilities import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
import torchvision.transforms.functional as F

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt


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
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models, help='model name')
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
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
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
    parser.add_argument("--ckpt_detection", type=str, default=None,
                        help='[path to detection model to produce bounding boxes')
    parser.add_argument("--multiscale_val", action='store_true', default=False,
                        help="enable multi-scale flipping inference during validation")
    parser.add_argument("--train_num_images", type=int, default=-1,
                        help="use fixed num of images for training (default: -1, i.e. use all available images")

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    # Visualization and logging options
    parser.add_argument("--enable_log", action='store_true', default=True,
                        help="use tensorboard for visualization adn logging")
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
        train_dst = VOCSegmentation(root=opts.data_root, year=opts.year, image_set='train', download=opts.download,
                                    transform=train_transform, return_bbox=opts.return_bbox,
                                    ckpt_detection=opts.ckpt_detection)
        val_dst = VOCSegmentation(root=opts.data_root, year=opts.year, image_set='val', download=False,
                                  transform=val_transform, return_bbox=opts.return_bbox,
                                  ckpt_detection=opts.ckpt_detection)

    if opts.dataset == 'cityscapes':
        train_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = Cityscapes(root=opts.data_root,
                               split='train', transform=train_transform)
        val_dst = Cityscapes(root=opts.data_root,
                             split='val', transform=val_transform)
    return train_dst, val_dst


def validate(opts, model, loader, device, metrics, criterion, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utilities.Denormalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
        img_id = 0

    multiscale = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75] if opts.multiscale_val else [1.0]
    flipping = [0, 1] if opts.multiscale_val else [0]
    loss_val = 0.0
    with torch.no_grad():
        for i, (images, labels, bboxes) in tqdm(enumerate(loader)):
            multi_avg = torch.zeros(images.shape[0], opts.num_classes, images.shape[2], images.shape[3], dtype=torch.float32)  # BxCxHxW
            for (scale, flip) in itertools.product(multiscale, flipping):
                img_s, labels_s, bboxes_s = et.ExtScale(scale=scale, is_tensor=True)(images, labels, bboxes)
                img_s, labels_s, bboxes_s = et.ExtRandomHorizontalFlip(p=flip)(img_s, labels_s, bboxes_s)

                img_s = img_s.to(device, dtype=torch.float32)
                bboxes_s = bboxes_s.to(device, dtype=torch.float32)
                labels_s = labels_s.to(device, dtype=torch.long)

                if opts.return_bbox:
                    outputs = model(img_s, bboxes_s)
                    bboxes_s.cpu()
                else:
                    outputs = model(img_s)
                loss_val += criterion(outputs, labels_s)  # just for the stats
                outputs = outputs.cpu()
                if flip:
                    outputs = F.hflip(outputs)
                outputs = F.resize(outputs, size=multi_avg.shape[2:])
                multi_avg += outputs
            preds = multi_avg.argmax(dim=1)
            if not opts.crop_val:
                # it is required to resize prediction to initial size in order to get correct iou
                # batch size = 1 in this case
                images, preds, _ = et.ExtResize(size=labels.shape[1:])(images, preds, bboxes)
            preds = preds.numpy()
            labels = labels.numpy().astype(np.uint8)

            metrics.update(labels, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), labels[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = labels[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save(f'results/{img_id}_image.png')
                    Image.fromarray(target).save(f'results/{img_id}_target.png')
                    Image.fromarray(pred).save(f'results/{img_id}_pred.png')

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig(f'results/{img_id}_overlay.png', bbox_inches='tight', pad_inches=0)
                    plt.close()
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

    # Setup logging and visualization
    if opts.enable_log:
        curr_time = datetime.now().strftime("%d-%m-%Y_%H-%M")
        tb_train = SummaryWriter(f'logs/{curr_time}/train')
        tb_val = SummaryWriter(f'logs/{curr_time}/val')

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    if opts.dataset == 'voc' and not opts.crop_val:
        opts.val_batch_size = 1

    train_dst, val_dst = get_dataset(opts)
    if opts.train_num_images > 0:
        train_dst, _ = data.random_split(train_dst, [opts.train_num_images, len(train_dst) - opts.train_num_images])
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2,
        drop_last=True)  # drop_last=True to ignore single-image batches.
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
    print(f'Dataset: {opts.dataset}, Train set: {len(train_dst)}, Val set: {len(val_dst)}')

    # Set up model (all models are constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utilities.set_bn_momentum(model.backbone, momentum=0.01)

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    if opts.return_bbox:
        optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr,
                                    momentum=0.9, weight_decay=opts.weight_decay)
    else:
        optimizer = torch.optim.SGD(params=[
            {'params': model.backbone.parameters(), 'lr': opts.lr},
            {'params': model.classifier.parameters(), 'lr': 10*opts.lr},
        ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    if opts.lr_policy == 'poly':
        scheduler = utilities.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion,
    # criterion = utilities.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utilities.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print(f'Model saved as {path}')

    utilities.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print(f'Training state restored from {opts.ckpt}')
        print(f'Model restored from {opts.ckpt}')
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    # ==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_log else None  # sample idxs for visualization
    denorm = utilities.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.test_only:
        model.eval()
        val_score, ret_samples, loss_val = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, criterion=criterion,
            ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        return

    interval_loss = 0
    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        for (images, labels, bboxes) in train_loader:
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            if opts.return_bbox:
                bboxes = bboxes.to(device, dtype=torch.float32)
                outputs = model(images, bboxes)
            else:
                outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            interval_loss += loss.detach().cpu().numpy()

            if cur_itrs % 10 == 0:
                interval_loss = interval_loss / 10
                print(f'Epoch {cur_epochs}, Itrs {cur_itrs}/{opts.total_itrs}, Loss={interval_loss:.5f}')
                if opts.enable_log:
                    tb_train.add_scalar('Loss', interval_loss, cur_itrs)
                interval_loss = 0.0

            if cur_itrs % opts.val_interval == 0:
                save_ckpt(f'checkpoints/latest_{opts.model}_{opts.dataset}_num_{len(train_dst)}.pth')
                print("validation...")
                model.eval()
                val_score, ret_samples, loss_val = validate(
                    opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, criterion=criterion,
                    ret_samples_ids=vis_sample_id)
                print(metrics.to_str(val_score))
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt(f'checkpoints/best_{opts.model}_{opts.dataset}_num_{len(train_dst)}.pth')

                if opts.enable_log:  # visualize validation score and samples
                    tb_val.add_scalar('Loss', loss_val, cur_itrs)
                    tb_val.add_scalar('[Val] Overall Acc', val_score['Overall Acc'], cur_itrs)
                    tb_val.add_scalar('[Val] Mean IoU', val_score['Mean IoU'], cur_itrs)

                    for k, (img, target, lbl) in enumerate(ret_samples):
                        img = (denorm(img) * 255).astype(np.uint8)
                        target = val_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                        lbl = val_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                        concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                        tb_val.add_image(f'Image and prediction {k}', torch.from_numpy(concat_img))

                model.train()
            scheduler.step()

            if cur_itrs >= opts.total_itrs:
                tb_train.close()
                tb_val.close()
                return


if __name__ == '__main__':
    main()
