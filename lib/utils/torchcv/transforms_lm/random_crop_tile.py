'''This random crop strategy is described in paper:
   [1] SSD: Single Shot MultiBox Detector
'''
import math
import torch
import random

from PIL import Image
# from torchcv.utils.box import box_iou, box_clamp
from ..box import box_iou, box_clamp


def random_crop_tile_lm(
        img, boxes, labels, linemap,
        scale_range=[0.8, 1],
        max_aspect_ratio=2.):
    '''Randomly crop a PIL image.

    Args:
      img: (PIL.Image) image.
      boxes: (tensor) bounding boxes, sized [#obj, 4].
      labels: (tensor) bounding box labels, sized [#obj,].
      scale_range: [float,float] minimal image width/height scale.
      max_aspect_ratio: (float) maximum width/height aspect ratio.

    Returns:
      img: (PIL.Image) cropped image.
      boxes: (tensor) object boxes.
      labels: (tensor) object labels.
    '''
    imw, imh = img.size

    scale = random.uniform(scale_range[0], scale_range[1])
    aspect_ratio = random.uniform(
        max(1 / max_aspect_ratio, scale * scale),
        min(max_aspect_ratio, 1 / (scale * scale)))
    w = int(imw * scale * math.sqrt(aspect_ratio))
    h = int(imh * scale / math.sqrt(aspect_ratio))

    x = random.randrange(imw - w)
    y = random.randrange(imh - h)

    img = img.crop((x, y, x + w, y + h))
    linemap = linemap.crop((x, y, x + w, y + h))

    center = (boxes[:, :2] + boxes[:, 2:]) / 2
    mask = (center[:, 0] >= x) & (center[:, 0] <= x + w) \
           & (center[:, 1] >= y) & (center[:, 1] <= y + h)
    if mask.any():
        boxes = boxes[mask] - torch.tensor([x, y, x, y], dtype=torch.float)
        boxes = box_clamp(boxes, 0, 0, w, h)
        labels = labels[mask]
    else:
        boxes = torch.tensor([[0, 0, 0, 0]], dtype=torch.float)
        labels = torch.tensor([0], dtype=torch.long)
    return img, boxes, labels, linemap
