import math
import torch
import random

from PIL import Image
from ..box import box_iou, box_clamp


def crop_box_lm(img, boxes, labels, linemap, box):
    x, y, x2, y2 = box
    w = x2 - x
    h = y2 - y
    img = img.crop((x, y, x2, y2))
    linemap = linemap.crop((x, y, x2, y2))

    # check if center is still inside tile_box, otherwise ignore box
    # (if center is not inside tile box, not possible to get IoU >= 0.5 --> treated as background anyways)
    center = (boxes[:, :2] + boxes[:, 2:]) / 2
    mask = (center[:, 0] >= x) & (center[:, 0] <= x2) & (center[:, 1] >= y) & (center[:, 1] <= y2)
    if mask.any():
        boxes = boxes[mask] - torch.tensor([x, y, x, y], dtype=torch.float)
        boxes = box_clamp(boxes, 0, 0, w, h)
        labels = labels[mask]
    else:
        boxes = torch.tensor([[0, 0, 0, 0]], dtype=torch.float)
        labels = torch.tensor([0], dtype=torch.long)
    return img, boxes, labels, linemap
