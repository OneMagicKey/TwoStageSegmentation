import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
import cv2 as cv
import numpy as np


def box_cxcywh_to_xyxy(x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=out_bbox.device)
    return b.to(dtype=torch.int16)


def detr_pred_to_bbox(pred, result_img_size=(513, 513), num_classes=21, conf=0.7):
    """ This funtion converts raw detr output to bbox mask tensor """
    bbox = torch.zeros((pred['pred_logits'].shape[0], num_classes, *result_img_size), dtype=torch.float32, device=pred['pred_logits'].device)
    bbox[:, 0, :, :] = 1

    probas = pred['pred_logits'].softmax(-1)[:, :, :-1]
    keep = probas.max(-1).values > conf

    for i in range(probas.shape[0]):
        image_bboxes = rescale_bboxes(pred['pred_boxes'][i, keep[i]], result_img_size)
        image_probs = probas[i, keep[i]]

        for class_probs, (x_min, y_min, x_max, y_max) in zip(image_probs, image_bboxes.tolist()):
            class_index = class_probs.argmax()
            bbox[i, class_index, y_min:y_max, x_min:x_max] = 1
            bbox[i, 0, y_min:y_max, x_min:x_max] = 0

    return bbox


def yolo_pred_to_bbox(predictions, input_img_size, result_img_size=(513, 513), num_classes=21, conf=0):
    bbox = torch.zeros((num_classes, *input_img_size))
    bbox[0, :, :] = 1

    for pred in predictions:
        if pred[4] > conf:  # confidence threshold
            # pred_conf = pred[4]
            (x_min, y_min, x_max, y_max, _, class_index) = pred.int() # cast to int so conf level equals to 0
            bbox[class_index + 1, y_min:y_max, x_min:x_max] = 1
            bbox[0, y_min:y_max, x_min:x_max] = 0

    bbox = F.resize(bbox, size=result_img_size, interpolation=T.InterpolationMode.NEAREST)
    bbox.to(torch.float32)
    return bbox


def masks_to_bboxes(input_mask, num_classes=21):
    """ Generate bboxes by segmentation masks """
    bbox = torch.zeros((input_mask.shape[0], num_classes, *input_mask.shape[1:]), dtype=torch.float)
    bbox[:, 0, :, :] = 1

    for class_index in range(1, num_classes):
        batch, y, x = torch.where(input_mask == class_index)

        class_masks = np.zeros_like(input_mask, dtype=np.uint8)
        class_masks[batch, y, x] = 1

        for batch_index, class_mask in enumerate(class_masks):
            contours, _ = cv.findContours(class_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                (x,y,w,h) = cv.boundingRect(contour)
                bbox[batch_index, class_index, y:y+h, x:x+w] = 1
                bbox[batch_index, 0, y:y+h, x:x+w] = 0

    return bbox
