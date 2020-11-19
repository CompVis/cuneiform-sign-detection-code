import numbers
import numpy as np
from PIL import Image
from PIL import ImageOps
import random
from random import randint

import torch.functional as F
# from skimage.transform import warp, AffineTransform

from bbox_utils import intersection_over_union


# own stuff

def convert2binaryPIL(lbl_ind):
    # convert to PIL binary '1' without dither
    lbl_im = Image.fromarray(np.uint8(lbl_ind))
    fn = (lambda x: 255 if x > 0 else 0)
    lbl_im = lbl_im.convert('L').point(fn, mode='1')
    return lbl_im

def pad2square(bb, context_pad_ratio=0, context_pad=0, take_long_side=True):
    # -- extract square patches using ground truth bounding boxes
    # assert (context_pad >= 0 and context_pad_ratio == 0) or (context_pad_ratio >= 0 and context_pad == 0)
    width = bb[2] - bb[0]
    height = bb[3] - bb[1]
    diff = width - height
    width_is_smaller = 0 > diff
    height_is_smaller = 0 < diff
    if take_long_side:
        # take long side
        if context_pad == 0:
            if width_is_smaller:
                context_pad = np.round(context_pad_ratio * height)
            else:
                context_pad = np.round(context_pad_ratio * width)
        bb[0] = bb[0] - context_pad - (width_is_smaller * np.ceil(0.5 * (height - width)))
        bb[2] = bb[2] + context_pad + (width_is_smaller * np.floor(0.5 * (height - width)))
        bb[1] = bb[1] - context_pad - (height_is_smaller * np.ceil(0.5 * (width - height)))
        bb[3] = bb[3] + context_pad + (height_is_smaller * np.floor(0.5 * (width - height)))
    else:
        # take small side
        if context_pad == 0:
            if width_is_smaller:
                context_pad = np.round(context_pad_ratio * width)
            else:
                context_pad = np.round(context_pad_ratio * height)
        bb[0] = bb[0] - context_pad - (height_is_smaller * np.ceil(0.5 * (height - width)))
        bb[2] = bb[2] + context_pad + (height_is_smaller * np.floor(0.5 * (height - width)))
        bb[1] = bb[1] - context_pad - (width_is_smaller * np.ceil(0.5 * (width - height)))
        bb[3] = bb[3] + context_pad + (width_is_smaller * np.floor(0.5 * (width - height)))

    return bb


# BBOX sampling / cropping functions

def crop_image(im, bb, context_pad=0, pad_to_square=False, mean_values=[0, 0, 0]):
    """
    Crop a window from the image for detection. Include surrounding context
    according to the `context_pad` configuration. Creates square crop which
    respects the aspect ratio.

    window: bounding box coordinates as xmin, ymin, xmax, ymax.
    """

    # copy list and use as ndarray
    bb = np.array(bb, dtype=int)  # list(bb)

    imw, imh = im.shape[:2]

    # pad to square while preserving aspect ratio
    if pad_to_square:
        bb = pad2square(bb, context_pad=context_pad)

        # -- check whether bbox inside image
        # pad: [x_min, y_min, x_max, y_max]

        pad = [0, 0, 0, 0]
        if (bb[0] < 0):
            pad[0] = abs(bb[0])
            bb[0] = 0
        if (bb[1] < 0):
            pad[1] = abs(bb[1])
            bb[1] = 0
        if (bb[2] > imh):
            pad[2] = bb[2] - imh
            bb[2] = imh
        if (bb[3] > imw):
            pad[3] = bb[3] - imw
            bb[3] = imw

        # -- apply zero padding if necessary
        im = im[bb[1]:bb[3], bb[0]:bb[2], :]

        channel_mean = np.reshape(mean_values, (1, 1, 3)).astype(np.uint8)
        if pad[0]>0:
            pad_left = np.multiply(np.ones(shape=(imw, pad[0], 3), dtype=np.uint8),
                                   np.tile(channel_mean,(imw, pad[0],1)))
            im = np.concatenate((pad_left, im), axis=1)
        if pad[1]>0:
            pad_up = np.multiply(np.ones(shape=(pad[1], imh, 3), dtype=np.uint8),
                                 np.tile(channel_mean, (pad[1], imh, 1)))
            im = np.concatenate((pad_up, im), axis=0)
        if pad[2]>0:
            pad_right = np.multiply(np.ones(shape=(imw, pad[2], 3), dtype=np.uint8),
                                    np.tile(channel_mean, (imw, pad[2], 1)))
            im = np.concatenate((im, pad_right), axis=1)
        if pad[3]>0:
            pad_down = np.multiply(np.ones(shape=(pad[3], imh, 3), dtype=np.uint8),
                                   np.tile(channel_mean, (pad[3], imh, 1)))
            im = np.concatenate((im, pad_down), axis=0)

        return im, bb.tolist()
    else:
        if context_pad > 0:
            # better use crop_pil_image
            return NotImplemented
        # return simple crop
        return im[bb[1]:bb[3], bb[0]:bb[2], :]


def crop_pil_image(im, bb, context_pad=0, pad_to_square=False, fill_values=None):
    """
    Crop a window from the image for detection. Include surrounding context
    according to the `context_pad` configuration. Creates square crop which
    respects the aspect ratio.

    window: bounding box coordinates as xmin, ymin, xmax, ymax.
    """

    # copy list and use as ndarray
    bb = np.array(bb, dtype=int)  # list(bb)

    imw, imh = im.size

    # pad to square while preserving aspect ratio
    if pad_to_square:
        bb = pad2square(bb, context_pad=context_pad)

        if fill_values is None:

            # if cropped out of image range, pillow pads with zeros automatically
            im = im.crop((bb[0], bb[1], bb[2], bb[3]))

        else:
            # check whether bbox inside image
            # pad: [x_min, y_min, x_max, y_max]
            pad = [0, 0, 0, 0]
            if bb[0] < 0:
                pad[0] = abs(bb[0])
                bb[0] = 0
            if bb[1] < 0:
                pad[1] = abs(bb[1])
                bb[1] = 0
            if bb[2] > imh:
                pad[2] = bb[2] - imh
                bb[2] = imh
            if bb[3] > imw:
                pad[3] = bb[3] - imw
                bb[3] = imw

            # crop box
            im = im.crop((bb[0], bb[1], bb[2], bb[3]))
            # apply zero padding if necessary
            im = ImageOps.expand(im, border=(pad[0], pad[1], pad[2], pad[3]), fill=tuple(fill_values))

        return im, bb.tolist()
    else:
        if context_pad > 0:
            bb[0] = max(bb[0] - context_pad, 0)
            bb[2] = min(bb[2] + context_pad, imw)
            bb[1] = max(bb[1] - context_pad, 0)
            bb[3] = min(bb[3] + context_pad, imh)
        # return simple crop
        return im.crop((bb[0], bb[1], bb[2], bb[3])), bb.tolist()


def spatial_sample(im_pad, bb, spatial_sample_rng, rnd_scale_ratio=0.05):
    im = im_pad
    imh, imw = im.shape[:2]
    im_bb = [0, 0, imw, imh]

    # make ground truth box square, and use its dimensions
    bb_gt = list(bb)
    w = bb[2] - bb[0]
    h = bb[3] - bb[1]
    if w > h:
        bb_gt[1] = int(bb_gt[1] - np.ceil(0.5 * (w - h)))
        bb_gt[3] = int(bb_gt[3] + np.floor(0.5 * (w - h)))
        h = w
    else:
        bb_gt[0] = int(bb_gt[0] - np.ceil(0.5 * (h - w)))
        bb_gt[2] = int(bb_gt[2] + np.floor(0.5 * (h - w)))
        w = h
    # add random scaling to test bbox
    # by treating dimension differently the aspect ratio will fluctuate a little (due to resizing afterwards!)
    wrange = round(rnd_scale_ratio * w)
    hrange = round(rnd_scale_ratio * h)
    w = min(w + random.randint(-wrange, 2*wrange), imw - 1)  # ensure size is in im_pad
    h = w  # min(h + random.randint(hrange, 2*hrange), imh - 1)  # ensure size is in im_pad

    # set ranges according to provided label
    min_IoU = spatial_sample_rng[0]
    max_IoU = spatial_sample_rng[1]

    max_iter = 500
    curr_iter = 0
    ratio = 0.0
    while curr_iter < max_iter and (ratio >= max_IoU or ratio <= min_IoU):
        curr_iter += 1

        # bbox sampling
        jxy = [randint(0, im_bb[2] - w), randint(0, im_bb[3] - h)]
        bb_test = list([jxy[0], jxy[1], w + jxy[0], h + jxy[1]])

        # check if new box fits criteria
        if min(bb_test) >= 0 and bb_test[2] <= im.shape[1] and bb_test[3] <= im.shape[0]:
            ratio = intersection_over_union(bb_test, bb_gt)

    if max_IoU >= ratio >= min_IoU:
        im = im[bb_test[1]:bb_test[3], bb_test[0]:bb_test[2], :]
        # new_bb_gt = [bb_gt[0] - bb_test[0], bb_gt[1] - bb_test[1],  bb_gt[2] - bb_test[0], bb_gt[3] - bb_test[1]]
        new_bb_gt = bb_gt
    else:
        im = im
        new_bb_gt = bb_gt

        # DEBUG_MODE = False
        # if DEBUG_MODE:
        #     print "tricky box", w, h, imw, imh

    return im, new_bb_gt, bb_test



# TRANSFORMS


class MyRandomZoom(object):
    def __init__(self, scale_range, interpolation=Image.BILINEAR):
        self.scale_range = scale_range
        self.interpolation = interpolation

    def __call__(self, img):
        scale = np.random.uniform(*self.scale_range)
        new_size = (int(img.height * scale), int(img.width * scale))
        return F.resize(img, new_size, self.interpolation)


class MyFuzzyZoom(object):
    """
    :param target_size: (2-tuple) height, width
    :param scale_range: (2-tuple) range from which target_size may deviate
    :param interpolation: ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional)
    """
    def __init__(self, target_size, scale_range, interpolation=Image.BILINEAR):

        self.target_size = target_size
        self.scale_range = scale_range
        self.interpolation = interpolation

    @staticmethod
    def get_params(scale_range):
        return np.random.uniform(*scale_range)

    def __call__(self, img):
        scale = self.get_params(self.scale_range)
        new_size = (int(self.target_size[0] * scale), int(self.target_size[1] * scale))
        return F.resize(img, new_size, self.interpolation)


class MyRandomChoiceZoom(object):
    def __init__(self, scales, p=None, interpolation=Image.BILINEAR):
        self.scales = scales
        self.interpolation = interpolation
        self.p = p

    def __call__(self, img):
        scale = np.random.choice(self.scales, replace=True, p=self.p)
        new_size = (int(img.height * scale), int(img.width * scale))
        return F.resize(img, new_size, self.interpolation)


class MyRandomCenteredRotation(object):
    """
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        translation_range (2-tuple): Range of pixels to select from.
            The center of rotation is shifted according to a number sampled from this range.
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
    """
    def __init__(self, degrees, translation_range=(-3, 3), resample=Image.BILINEAR):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees
        self.translation_range = translation_range
        self.resample = resample

    def __call__(self, img):

        angle = np.random.uniform(*self.degrees)
        translated_center = None
        if self.translation_range:
            translated_center = (
                np.random.uniform(*self.translation_range) + int(img.height/2),
                np.random.uniform(*self.translation_range) + int(img.width/2)
            )
        return F.rotate(img, angle, resample=self.resample, expand=False, center=translated_center)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


