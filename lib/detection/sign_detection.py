import torch
from torch.autograd import Variable
import torchvision
from torchvision.transforms import Resize, FiveCrop, CenterCrop

from ..utils.transform_utils import crop_pil_image


def crop_segment_from_tablet_im(pil_im, seg_bbox, context_pad_frac=0):
    """

    :param pil_im: full tablet image to crop from
    :param seg_bbox: bbox coordinates [xmin, ymin, xmax, ymax]
    :param context_pad_frac: the fraction of the minimum side length of bbox to use as padding
    :return: cropped segment as pil image
    """
    min_side = min((seg_bbox[2] - seg_bbox[0], seg_bbox[3]-seg_bbox[1]))
    context_pad = min_side * context_pad_frac
    # crop segment
    segment_crop, new_bbox = crop_pil_image(pil_im, seg_bbox, context_pad=context_pad, pad_to_square=False)
    return segment_crop, new_bbox


def rescale_segment_single(pil_im, scale):
    """ Produce PIL image of segment at selected scale

    :param pil_im: tablet segment that is to be processed
    :param scale: scale used for resizing
    :return: PIL image
    """
    # compute scaled size
    imw, imh = pil_im.size
    imw = int(imw * scale)
    imh = int(imh * scale)
    # compose transforms
    tablet_transform = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda x: x.convert('L')),  # convert to gray
            Resize((imh, imw)),  # resize according to scale
        ])
    # apply transforms
    input_im = tablet_transform(pil_im)
    return input_im


def preprocess_segment_single(pil_im, scale):
    """ produce tensor of segment at selected scale

    :param pil_im: tablet segment that is to be processed
    :param scale: scale used for resizing
    :return: 4D tensors
    """
    # compute scaled size
    imw, imh = pil_im.size
    imw = int(imw * scale)
    imh = int(imh * scale)
    # compose transforms
    tablet_transform = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda x: x.convert('L')),  # convert to gray
            Resize((imh, imw)),  # resize according to scale
            # tensor-space transforms
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5], std=[1]),  # normalize
        ])
    # apply transforms
    input_tensor = tablet_transform(pil_im).unsqueeze(0)
    return input_tensor


def preprocess_segment_multi_scale(pil_im, scales):
    """ produces multiple copies of the segment at different scales

    :param pil_im: tablet segment that is to be processed
    :param scales: list of scales
    :return: list of 3D tensors with different shapes (according to scales)
    """
    # compute scaled size
    imw, imh = pil_im.size
    # tensor-space transforms
    ts_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5], std=[1]),  # normalize
    ])
    # compose transforms
    tablet_transform = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda x: x.convert('L')),  # convert to gray
            # resize according to scales
            torchvision.transforms.Lambda(
                lambda crop: [Resize((int(imh * scale), int(imw * scale)))(crop) for scale in scales]),
            torchvision.transforms.Lambda(
                lambda scaled_crops: [ts_transform(crop) for crop in scaled_crops]),  # returns a 4D tensor
        ])
    # apply transforms
    im_list = tablet_transform(pil_im)
    return im_list


def preprocess_segment_for_eval(pil_im, scale, shift=0):
    """ produces five copies of the segment at slightly different offsets

    :param pil_im: tablet segment that is to be processed
    :param scale: scale which should be used for resizing
    :param shift: offset shift used to produce five-fold oversampling
    :return: 4D tensor with 5xCxWxH
    """
    # compute scaled size
    imw, imh = pil_im.size
    imw = int(imw * scale)
    imh = int(imh * scale)
    # determine crop size
    crop_sz = [int(imh - shift), int(imw - shift)]
    # tensor-space transforms
    ts_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5], std=[1]),  # normalize
    ])
    # compose transforms
    tablet_transform = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda x: x.convert('L')),  # convert to gray
            Resize((imh, imw)),  # resize according to scale
            FiveCrop((crop_sz[0], crop_sz[1])),  # oversample
            torchvision.transforms.Lambda(
                lambda crops: torch.stack([ts_transform(crop) for crop in crops])),  # returns a 4D tensor
        ])
    # apply transforms
    input_tensor = tablet_transform(pil_im)
    return input_tensor


def predict_im_list(model, im_list, use_gpu, min_sz=227):
    """ applies model to list of 3D tensors (unlike a 4D tensor in predict())

    :param model: network module that is used for the prediction
    :param im_list: list of 3D tensors
    :param use_gpu: boolean that indicates whether GPU is available
    :param min_sz: minimum side length of input
    :return: list of result tensors
    """
    # apply network model
    outputs = []
    for in_im in im_list:
        if (in_im.shape[1] >= min_sz) and (in_im.shape[2] >= min_sz):
            # prepare input
            if use_gpu:
                in_var = Variable(in_im.cuda(), volatile=True)  # volatile=True -> faster, less memory usage
            else:
                in_var = Variable(in_im, volatile=True)
            output = model(in_var.unsqueeze(0))
            outputs.append(output.data.cpu().numpy())
        else:
            outputs.append(None)

    # convert to numpy
    return outputs


def predict(model, inputs, use_gpu, use_bbox_reg=False):
    """ applies model to 4D tensor (batch of images)

    :param model: network module that is used for the prediction
    :param inputs: 4D tensor (batch of images)
    :param use_gpu: boolean that indicates whether GPU is available
    :param use_bbox_reg: boolean that indicates whether to use bbox regression
    :return: result tensor
    """
    # prepare input
    if use_gpu:
        inputs = Variable(inputs.cuda(), volatile=True)  # volatile=True -> faster, less memory usage
    else:
        inputs = Variable(inputs, volatile=True)
    # apply network model
    # output = model(inputs) # consumes to much memory

    if use_bbox_reg:
        scores, bboxes = [], []
        for in_im in inputs:
            o1, o2 = model(in_im.unsqueeze(0))
            scores.append(o1)
            bboxes.append(o2)
        # concat and convert to numpy
        output = torch.cat(scores, dim=0)
        predicted = output.data.cpu().numpy()
        output = torch.cat(bboxes, dim=0)
        predicted_roi = output.data.cpu().numpy()
    else:
        scores = []
        for in_im in inputs:
            scores.append(model(in_im.unsqueeze(0)))
        # concat and convert to numpy
        output = torch.cat(scores, dim=0)
        predicted = output.data.cpu().numpy()
        predicted_roi = []

    return predicted, predicted_roi

