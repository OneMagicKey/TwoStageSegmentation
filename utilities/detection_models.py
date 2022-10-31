import torch
import network


def load_yolo(path_to_yolo):
    yolo = torch.hub.load('ultralytics/yolov5', 'custom', path_to_yolo)
    yolo.conf, yolo.iou = 0.2, 0.6
    yolo = yolo.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yolo.to(device)
    return yolo


def load_detr(path_to_detr):
    detr = torch.hub.load('facebookresearch/detr', 'detr_resnet101', pretrained=False, num_classes=21)
    detr.load_state_dict(torch.load(path_to_detr, map_location='cpu')['model'])
    detr = detr.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detr.to(device)
    return detr


def load_primary(path_to_primary, opts):
    m = 'deeplabv3plus_xception'
    primary_model = network.modeling.__dict__[m](num_classes=opts.num_classes, output_stride=opts.output_stride)
    network.convert_to_separable_conv(primary_model.classifier)
    checkpoint = torch.load(path_to_primary, map_location=torch.device('cpu'))
    primary_model.load_state_dict(checkpoint["model_state"])
    primary_model = primary_model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    primary_model.to(device)
    return primary_model
