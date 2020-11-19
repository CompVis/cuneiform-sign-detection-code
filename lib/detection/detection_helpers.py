import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm

import PIL.Image as Image
from ..utils.transform_utils import crop_pil_image

from ..evaluations.config import cfg
from ..utils.bbox_utils import clip_boxes


def visualize_net_output_single(im, predicted, cunei_id=30, num_classes=None, min_prob=0.95):
    # visualize output of single crop detections

    if num_classes is None:
        num_classes = cfg.TEST.NUM_CLASSES

    # cross-image products
    output = np.mean(predicted, axis=0)
    # cross-channel products
    lbl_ind = np.argmax(output, axis=0)

    ctr_crop = predicted[0, ...]

    plt.figure(figsize=(16, 24))
    plt.subplot(4, 2, 1)
    plt.imshow(im, cmap=cm.Greys_r)
    plt.title('input')
    plt.subplot(4, 2, 2)
    plt.imshow(ctr_crop.squeeze()[cunei_id, ...])
    plt.colorbar()
    plt.title('class #{}'.format(cunei_id))
    plt.subplot(4, 2, 3)
    cmap = plt.get_cmap('Paired')
    plt.imshow(lbl_ind, cmap=cmap, vmin=0, vmax=num_classes)
    plt.colorbar()
    plt.title('argmax class')
    plt.subplot(4, 2, 4)
    test = np.argmax(ctr_crop.squeeze(), axis=0)
    test[np.max(ctr_crop.squeeze(), axis=0) < min_prob] = 0
    plt.imshow(test, cmap=cmap, vmin=0, vmax=num_classes)
    plt.colorbar()
    plt.title('argmax class ( {} confidence)'.format(min_prob))


def _refine_detections(predicted):
    # single image product from center crop
    ctr_crop = predicted[4, ...]
    # cross-image products
    output = np.mean(predicted, axis=0)
    max_output = np.max(predicted, axis=0)
    uncertainty = np.var(predicted, axis=0)
    # cross-channel products
    lbl_ind = np.argmax(output, axis=0)
    average_unc = np.mean(uncertainty, axis=0)
    min_average_unc = np.min(average_unc)
    max_average_unc = np.max(average_unc)
    max_unc = np.max(uncertainty)

    # save products
    # sio.savemat('results/test_tablet_cuneiNet_{}_{}_scale_{}_{}.mat'.format(training_round, imageName, scale, negatives_used),
    #     {#'probs':ctr_crop,
    #      #'pred_labels': np.argmax(ctr_crop,axis=0),
    #      #'entropy': -np.sum(ctr_crop * np.log(ctr_crop)),
    #      'predicted': predicted,
    #      'avg_probs': output,
    #      'avg_unc': average_unc,
    #      'avg_pred_labels': lbl_ind})

    return ctr_crop, output, max_output, uncertainty, lbl_ind, average_unc, min_average_unc, max_average_unc, max_unc


def visualize_net_output(im, predicted, cunei_id=30, num_classes=None):
    # visualize output of 5 star crop detections

    if num_classes is None:
        num_classes = cfg.TEST.NUM_CLASSES

    ctr_crop, output, max_output, uncertainty, \
    lbl_ind, average_unc, min_average_unc, max_average_unc, max_unc = _refine_detections(predicted)

    plt.figure(figsize=(16, 24))
    plt.subplot(3, 2, 1)
    plt.imshow(im, cmap=cm.Greys_r)
    plt.title('input')
    plt.subplot(3, 2, 2)
    plt.imshow(ctr_crop.squeeze()[cunei_id, ...])
    plt.colorbar()
    plt.title('class #{}'.format(cunei_id))
    plt.subplot(3, 2, 3)
    cmap = plt.get_cmap('Paired')
    plt.imshow(np.argmax(ctr_crop.squeeze(), axis=0), cmap=cmap, vmin=0, vmax=num_classes)
    plt.colorbar()
    plt.title('argmax class')
    plt.subplot(3, 2, 4)
    test = np.argmax(ctr_crop.squeeze(), axis=0)
    test[np.max(ctr_crop.squeeze(), axis=0) < 0.95] = 0
    plt.imshow(test, cmap=cmap, vmin=0, vmax=num_classes)
    plt.colorbar()
    plt.title('argmax class (0.95 confidence)')
    #plt.subplot(4, 2, 5)
    #cmap = plt.get_cmap('Paired')
    #plt.imshow(lbl_ind, cmap=cmap, vmin=0, vmax=num_classes)
    #plt.colorbar()
    #plt.title('avg argmax class')
    plt.subplot(3, 2, 5)
    plt.imshow(average_unc, vmin=0, vmax=max_average_unc)
    plt.colorbar()
    plt.title('shift induced uncertainty')
    plt.subplot(3, 2, 6)
    # entropy
    plt.imshow(-np.sum(ctr_crop.squeeze() * np.log(ctr_crop.squeeze()), axis=0))
    plt.colorbar()
    plt.title('entropy')


def _im_to_pyra_coords(pyra, boxes):
    # boxes is N x 4 where each row is a box in the image specified
    # by [x1 y1 x2 y2].
    #
    # Output is a cell array where cell i holds the pyramid boxes
    # coming from the image box
    boxes = boxes - 1
    pyra_boxes = []
    for level in range(pyra['num_levels']):
        level_boxes = boxes * pyra['scales'][level]
        level_boxes = np.round(level_boxes / pyra['stride'])
        level_boxes = level_boxes
        # add padding
        level_boxes[:, 0] = level_boxes[:, 0] + pyra['padx']
        level_boxes[:, 2] = level_boxes[:, 2] + pyra['padx']
        level_boxes[:, 1] = level_boxes[:, 1] + pyra['pady']
        level_boxes[:, 3] = level_boxes[:, 3] + pyra['pady']
        pyra_boxes.append(level_boxes)
    return pyra_boxes


def _pyra_to_im_coords(pyra, boxes):
    # boxes is N x 5 where each row is a box in the format [x1 y1 x2 y2 pyra_level]
    # where (x1, y1) is the upper-left corner of the box in pyramid level pyra_level
    # and (x2, y2) is the lower-right corner of the box in pyramid level pyra_level
    # Assumes 1-based indexing.
    # pyramid to im scale factors for each scale

    scales = pyra['stride'] / pyra['scales'][0]

    # pyramid to im scale factors for each pyra level in boxes
    if len(scales.shape) > 0:
        scales = scales[boxes[:, -1]];

    # Remove padding from pyramid boxes
    boxes[:, 0] = boxes[:, 0] - pyra['padx']
    boxes[:, 2] = boxes[:, 2] - pyra['padx']
    boxes[:, 1] = boxes[:, 1] - pyra['pady']
    boxes[:, 3] = boxes[:, 3] - pyra['pady']

    im_boxes = boxes[:, :4] * scales
    return im_boxes


def _pyramid_patch_box(x1, y1, feat_map_sz, pyra, lvl_idx, opt='A'):
    # compute image patch box coordinates in original image
    # should also work for all features of one image at once
    # REQUIREMENTS:
    # position of feature in feature map:                   x1, y1
    # dimension of feature map:                             feat_map_sz
    # scale of input image (relative to original scale):    pyra, lvl_idx
    # stride that is determined by network architecture:    pyra

    # OPTION A
    if opt == 'A':
        boxes = np.array([x1 - 0.5, y1 - 0.5, x1 + 0.5, y1 + 0.5]).transpose([1, 0])
        boxes = np.concatenate([boxes, np.tile(lvl_idx, [len(x1), 1])], axis=1)
    # OPTION B - more accurate
    elif opt == 'B':
        x_step = (feat_map_sz[1] - 1) / float(feat_map_sz[1])
        y_step = (feat_map_sz[0] - 1) / float(feat_map_sz[0])

        boxes = np.array(
            [(x1 - 0.5) * x_step, (y1 - 0.5) * y_step, (x1 + 0.5) * x_step, (y1 + 0.5) * y_step]).transpose([1, 0])
        boxes = np.concatenate([boxes, np.tile(lvl_idx, [len(x1), 1])], axis=1)

    im_patch_box = np.floor(_pyra_to_im_coords(pyra, boxes))
    return im_patch_box


def _pyramid_rf_box(im_sz, im_patch_box, rf_size, scales, lvl_idx):
    # compute receptive field box coordinates in original image
    # (given patch_box coordinates in original image)
    # REQUIREMENTS:
    # receptive field size determined by network architecture:  rf_size [H, W]
    # original image size:                                      im_sz
    # patch box size in original image:                         im_patch_box
    # scale of input image relative to original image:          scales, lvl_idx

    scaled_rf_sz = rf_size / scales[lvl_idx]

    im_rf_box = np.zeros_like(im_patch_box)
    im_rf_box[:, 0] = im_patch_box[:, 0] - scaled_rf_sz[1] / 2.
    im_rf_box[:, 1] = im_patch_box[:, 1] - scaled_rf_sz[0] / 2.
    im_rf_box[:, 2] = im_patch_box[:, 2] + scaled_rf_sz[1] / 2.
    im_rf_box[:, 3] = im_patch_box[:, 3] + scaled_rf_sz[0] / 2.

    # should not be required!!
    # im_rf_box = clip_boxes(im_rf_box, im_sz)

    return np.round(im_rf_box)


def compute_bbox_grids(map_shape, im_shape, arch_type='alexnet'):
    # stride, offset and receptive field [H, W] come from external excel spreadsheet calculation
    if arch_type is 'alexnet':
        pyra = {'stride': 32, 'num_levels': 1, 'scales': np.array([1.0]),
                'padx': 0, 'pady': 0, 'offset': 113, 'rf_size': [227, 227]}  # [227 rf] 195, 227
    else:
        pyra = {'stride': 32, 'num_levels': 1, 'scales': np.array([1.0]),
                'padx': 0, 'pady': 0, 'offset': 112, 'rf_size': [224, 224]}  # [227 rf] 195, 227

        # pyra = {'stride': 16, 'num_levels': 1, 'scales': np.array([1.0]),
        #         'padx': 0, 'pady': 0, 'offset': 112, 'rf_size': [224, 224]}  # [227 rf] 195, 227

    # blobs in caffe and images in opencv are H,W formatted. This results in YX format
    # since all bounding boxes use XY convention, then accessing images or blobs this needs to be taken into account
    x = np.arange(0, map_shape[1])
    y = np.arange(0, map_shape[0])
    xv, yv = np.meshgrid(x, y, sparse=False, indexing='xy')

    # print lbl_ind.shape, im.shape, xv.shape

    # compute basic patch boxes
    # each score in the score map corresponds to a single non-overlapping box
    patch_boxes = _pyramid_patch_box(xv.flatten(), yv.flatten(), map_shape, pyra, 0, opt='A') + pyra[
        'offset']

    # compute receptive field sized boxes (overlapping)
    # due to the way pyramid_rf_box is implemented one stride needs to be subtracted from rf_size
    rf_sz = np.array(pyra['rf_size'], dtype=np.int)
    rf_boxes = _pyramid_rf_box(im_shape, patch_boxes, rf_sz - pyra['stride'], pyra['scales'], 0)

    return patch_boxes, rf_boxes


def label_map2image(feat_x, feat_y, map_shape, arch_type='alexnet'):
    # stride, offset and receptive field [H, W] come from external excel spreadsheet calculation
    if arch_type is 'alexnet':
        pyra = {'stride': 32, 'num_levels': 1, 'scales': np.array([1.0]),
                'padx': 0, 'pady': 0, 'offset': 113, 'rf_size': [227, 227]}  # [227 rf] 195, 227
    else:
        pyra = {'stride': 32, 'num_levels': 1, 'scales': np.array([1.0]),
                'padx': 0, 'pady': 0, 'offset': 112, 'rf_size': [224, 224]}  # [227 rf] 195, 227

    # compute basic patch boxes
    # each score in the score map corresponds to a single non-overlapping box
    patch_boxes = _pyramid_patch_box(feat_x, feat_y, map_shape, pyra, 0, opt='A') + pyra[
        'offset']

    return patch_boxes


def radius_in_image(feat_radius, dim=0, arch_type='alexnet'):
    # dim defines along which dimension to compute (only important, if rf_size not square)

    # stride, offset and receptive field [H, W] come from external excel spreadsheet calculation
    if arch_type is 'alexnet':
        pyra = {'stride': 32, 'num_levels': 1, 'scales': np.array([1.0]),
                'padx': 0, 'pady': 0, 'offset': 113, 'rf_size': [227, 227]}  # [227 rf] 195, 227
    else:
        pyra = {'stride': 32, 'num_levels': 1, 'scales': np.array([1.0]),
                'padx': 0, 'pady': 0, 'offset': 112, 'rf_size': [224, 224]}  # [227 rf] 195, 227

    rf_sz = np.array(pyra['rf_size'], dtype=np.int)

    # compute radii in image
    patch_radius = feat_radius * pyra['stride']
    rf_radius = patch_radius + (rf_sz[dim] - pyra['stride'])
    return patch_radius, rf_radius


def coord_in_image(coord, add_rf=False, arch_type='alexnet'):
    # stride, offset and receptive field [H, W] come from external excel spreadsheet calculation
    if arch_type is 'alexnet':
        pyra = {'stride': 32, 'num_levels': 1, 'scales': np.array([1.0]),
                'padx': 0, 'pady': 0, 'offset': 113, 'rf_size': [227, 227]}  # [227 rf] 195, 227
    else:
        pyra = {'stride': 32, 'num_levels': 1, 'scales': np.array([1.0]),
                'padx': 0, 'pady': 0, 'offset': 112, 'rf_size': [224, 224]}  # [227 rf] 195, 227

    rf_sz = np.array(pyra['rf_size'], dtype=np.int)

    # compute coordinate in image
    im_coord = coord * pyra['stride'] + pyra['offset']
    if add_rf:
        im_coord += (rf_sz[0] - pyra['stride'])
    return im_coord


def _bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes


def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def crop_bboxes_from_im(im, bboxes, context_pad=0, is_pil=False):
    """
    Crop a bbox from the image for detection.
    im: crop target
    bboxes: bounding box coordinates as xmin, ymin, xmax, ymax.
    """
    # iterate over boxes
    im_crop_list = []
    for i in xrange(bboxes.shape[0]):
        # format bbox
        bbox = np.round(bboxes[i, :]).astype(int)
        # crop bbox from image
        if context_pad <= 0:
            im_crop = im[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        else:
            if is_pil:
                im_crop = np.asarray(crop_pil_image(im, bbox, context_pad=context_pad)[0])
            else:
                im_crop = np.asarray(crop_pil_image(Image.fromarray(im), bbox, context_pad=context_pad)[0])  # Image.fromarray(im, 'L')
        # append to list
        im_crop_list.append(im_crop)
    return im_crop_list


def apply_bbox_regression(predicted_roi, rf_boxes, im_shape, num_classes=None, with_star_crop=True):
    if num_classes is None:
        num_classes = cfg.TEST.NUM_CLASSES

    ## if use_bbox_reg:
    # select roi deltas
    if with_star_crop:
        roi_deltas = predicted_roi[4, ...].reshape([num_classes * 4, -1]).transpose()
    else:
        roi_deltas = predicted_roi.reshape([num_classes * 4, -1]).transpose()
    # apply bounding-box regression deltas
    pred_boxes = _bbox_transform_inv(rf_boxes, roi_deltas)
    # make sure everything stays inside its limits
    pred_boxes = clip_boxes(pred_boxes, im_shape)
    return pred_boxes


def _split_detections(detections, boxes, axis=1, nsplits=2, sid=1):
    assert (axis == 1) | (axis == 2)
    boxes_split = np.array_split(boxes, nsplits, axis=axis-1)
    dets_split = np.array_split(detections, nsplits, axis=axis)
    # reshape to original format
    det_vec = dets_split[sid].reshape([dets_split[sid].shape[0], -1]).transpose()
    box_vec = boxes_split[sid].reshape([-1, boxes_split[sid].shape[-1]])
    return det_vec, box_vec


def split_detections(detections, pred_boxes, rf_boxes, lbl_map_shape,
                     split_axis='h', nsplits=2, sid=1, num_classes=cfg.TEST.NUM_CLASSES):
    if split_axis == 'h':
        # horizontal
        axis = 1
    elif split_axis == 'v':
        # vertical
        axis = 2
    # split detections
    if cfg.TEST.BBOX_REG:
        det_vec, box_vec = _split_detections(detections,
                                             pred_boxes.reshape(list(lbl_map_shape) + [4 * num_classes]),
                                             axis=axis, nsplits=nsplits, sid=sid)
    else:
        det_vec, box_vec = _split_detections(detections,
                                             rf_boxes.reshape(list(lbl_map_shape) + [num_classes]),
                                             axis=axis, nsplits=nsplits, sid=sid)

    return det_vec, box_vec


def vis_detections(im, bboxes, scores=None, labels=None, thresh=0.3, max_vis=20, figs_sz=(14, 14), ax=None):
    """
    Visualize bounding boxes on top of input image including labels / scores.
    im: input image
    bboxes: ndarray of bounding boxes
    scores: list of scores with length equal bboxes.shape[0]
    labels: list of integer labels with length equal bboxes.shape[0]
    etc.
    """
    if scores is None:
        nvis = min(max_vis, bboxes.shape[0])
    else:
        assert len(scores) == bboxes.shape[0]
        inds = np.where(scores > thresh)[0]
        nvis = min(max_vis, len(inds))

    # return if no bboxes to visualize
    if nvis == 0:
        return

    # plot base figure
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figs_sz)

    ax.imshow(im, cmap=cm.Greys_r)

    # iterate over bboxes and add them
    for i in xrange(nvis):
        bbox = bboxes[i, :4]
        # deal with scores
        if scores is not None:
            score = scores[i]
            # only show boxes with score above threshold
            if score <= thresh:
                continue
        # deal with labels
        if isinstance(labels, str):
            # if label is string
            cls_name = labels
            title_txt = labels
        else:
            # else assume index array
            assert len(labels) == bboxes.shape[0]
            cls_name = '{:.0f}'.format(labels[i])
            title_txt = 'X'

        # plt.cla()
        # plt.imshow(im, cmap = cm.Greys_r)
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='blue', alpha=0.5, linewidth=2.0)
        )
        if scores is None:
            ax.text(bbox[0], bbox[1] - 2, '{:s}'.format(cls_name),
                      bbox=dict(facecolor='blue', alpha=0.4), fontsize=8, color='white')
            ax.set_title('{}'.format(cls_name))
        else:
            ax.text(bbox[0], bbox[1] - 2, '{:s} {:.2f}'.format(cls_name, score),
                      bbox=dict(facecolor='blue', alpha=0.4), fontsize=8, color='white')
            ax.set_title('{}  {:.3f}'.format(cls_name, score))
        ax.set_title('{} detections with p({} | box) >= {:.2f}'.format(nvis, title_txt, thresh), fontsize=14)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
    # plt.axis('off')
    # plt.tight_layout()
    # plt.draw()


def scale_detection_boxes(boxes, scale_factor):
    # scale boxes depending on scale factor
    return boxes * scale_factor


def correct_for_shift(boxes, correction):
    # correct shift due to oversampling
    # in order to correct ground truth boxes, e.g. center crop -> subtract half the shift from gt boxes
    # in order to correct detection boxes, e.g. center crop -> add half the shift to detection boxes
    return boxes + correction


def reverse_scaling(rf_boxes, pred_boxes, scaling=1):
    # if used, should be applied right before post-processing detections

    # reverse scaling of detection boxes
    rf_boxes = scale_detection_boxes(rf_boxes, scaling)
    pred_boxes = scale_detection_boxes(np.array(pred_boxes), scaling)

    return rf_boxes, pred_boxes


def reverse_shift_and_scaling(rf_boxes, pred_boxes, shift=0, scaling=1):
    # if used, should be applied right before post-processing detections

    # correct shift of detection boxes due to center crop
    rf_boxes = correct_for_shift(rf_boxes, shift)
    pred_boxes = correct_for_shift(np.array(pred_boxes), shift)

    # reverse scaling of detection boxes
    rf_boxes = scale_detection_boxes(rf_boxes, scaling)
    pred_boxes = scale_detection_boxes(np.array(pred_boxes), scaling)

    return rf_boxes, pred_boxes


def post_process_detections(scores, pred_boxes, rf_boxes, num_classes=None, use_bbox_reg=None, nms_thresh=None):
    # apply nms and filter low confidence boxes
    # return list of good candidates
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    if num_classes is None:
        num_classes = cfg.TEST.NUM_CLASSES
    if use_bbox_reg is None:
        use_bbox_reg = cfg.TEST.BBOX_REG
    if nms_thresh is None:
        nms_thresh = cfg.TEST.NMS

    score_min_thresh = cfg.TEST.SCORE_MIN_THRESH
    score_bg_thresh = cfg.TEST.SCORE_BG_THRESH

    num_images = 1

    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]  # xrange vs range

    for i in range(num_images):  # xrange vs range
        # load image and get detections from network
        # [.....]
        # skip j = 0, because it's the background class
        for j in range(1, num_classes):  # xrange vs range
            # selection of boxes before NMS
            inds = np.where((scores[:, j] > score_min_thresh) & (scores[:, 0] < score_bg_thresh))[0]
            cls_scores = scores[inds, j]
            if use_bbox_reg:
                cls_boxes = pred_boxes[inds, j * 4:(j + 1) * 4]  # bbox regression
            else:
                cls_boxes = rf_boxes[inds, :]  # without bbox regression
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            # apply nms suppression
            keep = nms(cls_dets, nms_thresh)
            cls_dets = cls_dets[keep, :]
            all_boxes[j][i] = cls_dets

    return all_boxes


def get_all_bboxes(all_boxes):
    # take detections and all_boxes
    # return enriched list of detections including bbox, score, and max label
    num_classes = len(all_boxes)
    dets_list = [[] for _ in xrange(num_classes)]
    for j in xrange(1, num_classes):
        if len(all_boxes[j][0]) > 0:
            # get boxes
            BB = all_boxes[j][0]
            confidence = all_boxes[j][0][:, -1]
            # sort boxes by confidence
            sorted_ind = np.argsort(-confidence)
            # sorted_scores = np.sort(-confidence)
            BB = BB[sorted_ind, :]
            # append together with class label
            dets_list[j] = np.concatenate([BB, np.tile(j, reps=(BB.shape[0], 1))], axis=1)
    # concatenate lists from different classes
    return dets_list


def get_detection_bboxes(detections, all_boxes):
    # take detections and all_boxes
    # return enriched list of detections including bbox, score, and max label
    num_classes = len(all_boxes)
    dets_list = [[] for _ in xrange(num_classes)]
    for j in xrange(1, num_classes):
        if len(detections[j][0]) > 0:
            # get boxes
            BB = all_boxes[j][0]
            confidence = all_boxes[j][0][:, -1]
            # sort boxes by confidence
            sorted_ind = np.argsort(-confidence)
            # sorted_scores = np.sort(-confidence)
            BB = BB[sorted_ind, :]
            # select detections (indices require sorted detections - since sorted in evaluate)
            inds = detections[j][0]
            # append together with class label
            dets_list[j] = np.concatenate([BB[inds, :], np.tile(j, reps=(inds.shape[0], 1))], axis=1)
    # concatenate lists from different classes
    return dets_list


def collect_detection_crops(input_im, dets_list, max_vis=5, context_pad=0):
    # take tablet(input_im) and list of bboxes(dets_list)
    # return cropped patches(dets_crops)
    num_classes = len(dets_list)
    dets_crops = [[] for _ in xrange(num_classes)]
    for j in xrange(1, num_classes):
        if len(dets_list[j]) > 0:
            # get boxes
            cls_dets = dets_list[j]  # select class list
            bboxes = cls_dets[:, :4]  # remove any additional dims
            ncrops = min(max_vis, bboxes.shape[0])
            dets_crops[j] = crop_bboxes_from_im(input_im, bboxes[:ncrops, ...], context_pad)
    return dets_crops


def plot_crop_list(dets_crops, gt_crops, scores=None, k=8, cls_label='', figs_sz=(14, 4.5), context_pad=0):
    # plot co-detections of a single class
    # can handle dets_crops and gt_crops together or both on their own
    nvis = min(len(dets_crops), k)
    ngt = len(gt_crops)
    if nvis > 0:
        # slice crops and scores
        top_list = dets_crops[:nvis]
        top_vals = scores[:nvis]

        # prepare subplots (nvis or nvis + 1)
        fig, axes = plt.subplots(1, nvis + (ngt > 0), figsize=figs_sz, squeeze=False)  # , gridspec_kw={'wspace': 1}
        axes = axes.ravel()

        # plot idx
        pid = 0

        # plot ground truth in front if available
        if ngt > 0:
            axes[pid].imshow(gt_crops[0], cmap=cm.Greys_r)
            axes[pid].set_yticks([])
            axes[pid].set_xticks([])
            axes[pid].set_title("gt [{}]".format(cls_label))
            pid += 1

            # iterate over top_list
        for i, imcrop in enumerate(top_list):
            axes[pid + i].imshow(imcrop, cmap=cm.Greys_r)
            axes[pid + i].set_yticks([])
            axes[pid + i].set_xticks([])
            bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.8)
            # if there is no gt, add class label to title in first plot
            if pid + i == 0 and ngt == 0:
                axes[pid + i].set_title("class [{}] #{} p(x)={:.1f}".format(cls_label, i + 1, top_vals[i]))
            else:
                axes[pid + i].set_title("#{} p(x)={:.1f}".format(i + 1, top_vals[i]))

            if context_pad > 0:
                imw, imh = imcrop.shape[:2]
                bbox = [context_pad, context_pad, imh - context_pad, imw - context_pad]
                axes[pid + i].add_patch(plt.Rectangle((bbox[0], bbox[1]),
                                                      bbox[2] - bbox[0], bbox[3] - bbox[1],
                                                      fill=False, edgecolor='blue', linestyle='-',
                                                      alpha=0.3, linewidth=2.0))

    elif ngt > 0:
        nvis = 1
        # plot top k
        fig, axes = plt.subplots(1, nvis, figsize=figs_sz, squeeze=False)
        axes = axes.ravel()

        top_list = [gt_crops[0]] * nvis
        top_vals = [1] * len(top_list)
        for pid, imcrop in enumerate(top_list):
            axes[0, pid].imshow(imcrop, cmap=cm.Greys_r)
            bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.8)
            axes[0, pid].set_title("gt [{}]".format(cls_label))
            axes[0, pid].set_yticks([])
            axes[0, pid].set_xticks([])


def convert_detections_to_array(all_boxes, img_idx=0, idx_column=None):
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    total_labels = len(all_boxes)  # all_boxes.shape[0]
    temp = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # [ID, cx, cy, score, x1, y1, x2, y2, idx]
    detections_arr = np.zeros((0, 9))
    idx = 0
    # convert to CLS, cx, cy, score
    for i in range(total_labels):
        for box in all_boxes[i][img_idx]:
            temp[0] = i
            temp[1] = (box[2] + box[0]) / 2
            temp[2] = (box[3] + box[1]) / 2
            temp[3] = box[4]
            temp[4:8] = box[0:4]
            if idx_column is None:
                temp[8] = idx
            else:
                temp[8] = idx_column[idx]
            idx += 1
            detections_arr = np.vstack((detections_arr, temp))
    # SORT BY SCORE!?
    return detections_arr

