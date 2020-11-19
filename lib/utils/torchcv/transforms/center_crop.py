import torch


def center_crop(img, boxes, size):
    '''Crops the given PIL Image at the center.
    Args:
      img: (PIL.Image) image to be cropped.
      boxes: (tensor) object boxes, sized [#ojb,4].
      size (tuple): desired output size of (w,h).
    Returns:
      img: (PIL.Image) center cropped image.
      boxes: (tensor) center cropped boxes.
    '''
    w, h = img.size
    ow, oh = size
    i = int(round((h - oh) / 2.))
    j = int(round((w - ow) / 2.))
    img = img.crop((j, i, j + ow, i + oh))
    boxes -= torch.Tensor([j, i, j, i])
    boxes[:, 0::2].clamp(min=0, max=ow - 1)
    boxes[:, 1::2].clamp(min=0, max=oh - 1)
    return img, boxes
