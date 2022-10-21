import torch
import network


def load_yolo(path_to_yolo):
    yolo = torch.hub.load('ultralytics/yolov5', 'custom', path_to_yolo)
    # torch.hub.load('/content/yolov5', 'custom', path_to_yolo, source='local')
    yolo.conf, yolo.iou = 0.2, 0.6
    return yolo


def load_detr(path_to_detr):
    detr = torch.hub.load('facebookresearch/detr', 'detr_resnet101', pretrained=False, num_classes=21)
    detr.load_state_dict(torch.load(path_to_detr, map_location='cpu')['model'])
    return detr


def load_primary(path_to_primary, opts):
    m = 'deeplabv3plus_xception'
    primary_model = network.modeling.__dict__[m](num_classes=opts.num_classes, output_stride=opts.output_stride)
    network.convert_to_separable_conv(primary_model.classifier)
    checkpoint = torch.load(path_to_primary, map_location=torch.device('cpu'))
    primary_model.load_state_dict(checkpoint["model_state"])
    return primary_model
