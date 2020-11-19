import torch
import random

from PIL import Image


def pad_lm(img, linemap, target_size):
    '''Pad image with zeros to the specified size.

    Args:
      img: (PIL.Image) image to be padded.
      target_size: (tuple) target size of (ow,oh).

    Returns:
      img: (PIL.Image) padded image.

    Reference:
      `tf.image.pad_to_bounding_box`
    '''
    w, h = img.size
    canvas = Image.new('L', target_size)
    canvas.paste(img, (0, 0))  # paste on the left-up corner

    canvas_line = Image.new('1', target_size)
    canvas_line.paste(linemap, (0, 0))  # paste on the left-up corner
    return canvas, canvas_line
