# --------------------------------------------------------
# Parts of code adapted from Ross Girshick's Fast/er R-CNN code
# --------------------------------------------------------

import numpy as np


def unique_boxes(boxes, scale=1.0):
    """Return indices of unique boxes."""
    v = np.array([1, 1e3, 1e6, 1e9])
    hashes = np.round(boxes * scale).dot(v)
    _, index = np.unique(hashes, return_index=True)
    return np.sort(index)


def xywh_to_xyxy(boxes):
    """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
    return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))


def xyxy_to_xywh(boxes):
    """Convert [x1 y1 x2 y2] box format to [x y w h] format."""
    return np.hstack((boxes[:, 0:2], boxes[:, 2:4] - boxes[:, 0:2] + 1))


def convert_bbox_global2local(gbbox, seg_bbox):
    relative_bbox = np.array(gbbox) - np.array(seg_bbox[:2] * 2)
    return relative_bbox.tolist()


def convert_bbox_local2global(lbbox, seg_bbox):
    global_bbox = np.array(lbbox) + np.array(seg_bbox[:2] * 2)
    return global_bbox.tolist()


def validate_boxes(boxes, width=0, height=0):
    """Check that a set of boxes are valid."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    assert (x1 >= 0).all()
    assert (y1 >= 0).all()
    assert (x2 >= x1).all()
    assert (y2 >= y1).all()
    assert (x2 < width).all()
    assert (y2 < height).all()


def filter_small_boxes(boxes, min_size):
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    keep = np.where((w >= min_size) & (h > min_size))[0]
    return keep


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    usage for single: bb_new = clip_boxes(bb[np.newaxis, :], [imw, imh]).squeeze()
    """

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


# def clip_boxes(boxes, im_shape):
#     """Clip boxes to image boundaries."""
#     # x1 >= 0
#     boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
#     # y1 >= 0
#     boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
#     # x2 < im_shape[1]
#     boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
#     # y2 < im_shape[0]
#     boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
#     return boxes


def intersection_over_union(Reframe, GTframe):
    # by Oemer

    x1 = Reframe[0]
    y1 = Reframe[1]
    width1 = Reframe[2] - Reframe[0]
    height1 = Reframe[3] - Reframe[1]

    x2 = GTframe[0]
    y2 = GTframe[1]
    width2 = GTframe[2] - GTframe[0]
    height2 = GTframe[3] - GTframe[1]

    endx = max(x1 + width1, x2 + width2)
    startx = min(x1, x2)
    width = width1 + width2 - (endx - startx)

    endy = max(y1 + height1, y2 + height2)
    starty = min(y1, y2)
    height = height1 + height2 - (endy - starty)

    if width <= 0 or height <= 0:
        ratio = 0
    else:
        Area = width * height
        Area1 = width1 * height1
        Area2 = width2 * height2
        ratio = Area * 1. / (Area1 + Area2 - Area)
    # return IOU
    return ratio  # Reframe,GTframe


def bb_intersection_over_union(box_a, box_b):
    # adopted from https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    # compute the area of intersection rectangle
    inter_area = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    if (xB - xA + 1) <= 0 or (yB - yA + 1) <= 0:
        iou = 0
    else:
        iou = inter_area / float(box_a_area + box_b_area - inter_area)

    # return the intersection over union value
    return iou


def box_iou(box1, box2):
    '''Compute the intersection over union of two set of boxes.
    TD: modified to be legacy compatible
    The box order must be (xmin, ymin, xmax, ymax).
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    '''
    N = box1.shape[0]
    M = box2.shape[0]

    lt = np.maximum(box1[:,None,:2], box2[:,:2])  # [N,M,2]
    rb = np.minimum(box1[:,None,2:], box2[:,2:])  # [N,M,2]

    wh = np.clip((rb-lt+1.), 0, None)      # [N,M,2]
    inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

    area1 = (box1[:,2]-box1[:,0]+1.) * (box1[:,3]-box1[:,1]+1.)  # [N,]
    area2 = (box2[:,2]-box2[:,0]+1.) * (box2[:,3]-box2[:,1]+1.)  # [M,]
    iou = inter / (area1[:,None] + area2 - inter)
    return iou


def box_iou_org(box1, box2):
    '''Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    '''
    N = box1.shape[0]
    M = box2.shape[0]

    lt = np.maximum(box1[:,None,:2], box2[:,:2])  # [N,M,2]
    rb = np.minimum(box1[:,None,2:], box2[:,2:])  # [N,M,2]

    wh = np.clip((rb-lt), 0, None)      # [N,M,2]
    inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

    area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
    area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
    iou = inter / (area1[:,None] + area2 - inter)
    return iou


