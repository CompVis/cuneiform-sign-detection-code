import pandas as pd
import numpy as np
from ast import literal_eval

import os.path

from .config import cfg

from ..detection.detection_helpers import scale_detection_boxes, correct_for_shift, crop_bboxes_from_im

# class to wrap annotations


class BBoxAnnotations(object):

    def __init__(self, collection_name, relative_path='../'):
        # basic paths
        self.data_root = cfg.DATA_TEST_DIR
        self.num_classes = cfg.TEST.NUM_CLASSES
        self.path_to_data_products = '{}data/annotations/'.format(relative_path)

        # load collection annotations
        self.anno_df = self.load_collection_annotations(collection_name)

        if len(self.anno_df) > 0:
            print('Load bbox annotations for {} dataset: {} found!'.format(collection_name,
                                                                           self.anno_df.segm_idx.nunique()))
        else:
            print('No bbox annotations for {} dataset'.format(collection_name))

    def load_collection_annotations(self, collection_name):
        # assemble annotation file path
        annotation_file = 'bbox_annotations_{}.csv'.format(collection_name)
        annotation_file_path = '{}{}'.format(self.path_to_data_products, annotation_file)

        # check if annotation file exists
        if os.path.isfile(annotation_file_path):
            # read annotation file
            anno_df = pd.read_csv(annotation_file_path, engine='python')
            # convert string of list to list
            anno_df['relative_bbox'] = anno_df['relative_bbox'].apply(literal_eval)
            anno_df['bbox'] = anno_df['bbox'].apply(literal_eval)
            # return data frame
            return anno_df
        else:
            # return empty list (check later with len(.) to see if file exists)
            return []

    def select_anno_df_by_segm_idx(self, segm_idx):
        # wrap pandas logic
        return self.anno_df[(self.anno_df.segm_idx == segm_idx)]

    def select_anno_df_by_cdli_and_view(self, cdli, view):
        # wrap pandas logic
        return self.anno_df[(self.anno_df.tablet_CDLI == cdli) & (self.anno_df.view_desc == view)]


# static functions
def get_boxes_and_labels(anno_df):
    # retrieves gt_boxes and gt_labels from anno_df
    #gt_boxes = np.stack(anno_df.bbox.values)
    if len(anno_df) > 0:
        gt_boxes = np.stack(anno_df.relative_bbox.values)  # use relative bbox
        gt_labels = anno_df.train_label.values
    else:
        gt_boxes, gt_labels = np.array([]), np.array([])  # just dummy
    return gt_boxes, gt_labels


def get_class_gt_boxes(gt_boxes, gt_labels, cls_id):
    inds = np.where(gt_labels == cls_id)[0]
    return gt_boxes[inds, :]


def apply_scaling_and_shift(gt_boxes, scaling=1, shift=0):
    # if used, should be applied before calling eval
    # apply scaling of detection boxes
    gt_boxes = scale_detection_boxes(gt_boxes, scaling)
    # apply shift of detection boxes due to center crop
    gt_boxes = correct_for_shift(gt_boxes, shift)
    return gt_boxes


def apply_scaling(gt_boxes, scaling=1):
    # if used, should be applied before calling eval
    # apply scaling of detection boxes
    gt_boxes = scale_detection_boxes(gt_boxes, scaling)
    return gt_boxes


def collect_gt_crops(gt_boxes, gt_labels, im, num_classes, max_vis=2):
    # takes tablet image
    # returns list of ground truth crops organized by class
    gt_crops = [[] for _ in xrange(num_classes)]
    for j in xrange(1, num_classes):
        BBGT = get_class_gt_boxes(gt_boxes, gt_labels, j).astype(float)
        npos = BBGT.shape[0]
        if npos > 0:
            # get boxes
            bboxes = BBGT[:, :4]  # remove any additional dims
            ncrops = min(max_vis, bboxes.shape[0])
            gt_crops[j] = crop_bboxes_from_im(im, bboxes[:ncrops, :])
    return gt_crops


def prepare_segment_gt(segm_idx, segm_scale, bbox_anno, with_star_crop=False):
    # this is how things work together

    # create empty lists in case no annotations available
    gt_boxes, gt_labels = [], []

    if len(bbox_anno.anno_df) > 0:
        # select annotations for specific segment
        sub_anno_df = bbox_anno.select_anno_df_by_segm_idx(segm_idx)
        # get boxes and labels
        gt_boxes, gt_labels = get_boxes_and_labels(sub_anno_df)
        # adapt gt boxes to input format
        if with_star_crop:
            gt_boxes = apply_scaling_and_shift(gt_boxes, scaling=segm_scale, shift=-cfg.TEST.SHIFT / 2.)
        else:
            gt_boxes = apply_scaling(gt_boxes, scaling=segm_scale)

    # return selected ground truth
    return gt_boxes, gt_labels


# def get_pred_boxes_df(all_boxes, seg_idx):
#     # iterate list
#     list_boxes = []
#     list_cls_idx = []
#     for cls, boxes in enumerate(all_boxes):
#         num_boxes = len(boxes[0])
#         if num_boxes > 0:
#             list_boxes.append(boxes)
#             list_cls_idx.extend([cls] * num_boxes)
#     # create df
#     pred_boxes_df = pd.DataFrame()  # []
#     if len(list_boxes) > 0:
#         pred_boxes_df = pd.DataFrame(np.hstack(list_boxes).reshape(-1, 5), columns=['x1', 'y1', 'x2', 'y2', 'conf'])
#         pred_boxes_df['cls'] = list_cls_idx
#         pred_boxes_df['seg_idx'] = seg_idx
#
#     return pred_boxes_df
#
#
# def get_gt_boxes_df(gt_boxes, gt_labels, seg_idx):
#     # create df
#     gt_boxes_df = pd.DataFrame()  # []
#     if len(gt_boxes) > 0:
#         gt_boxes_df = pd.DataFrame(gt_boxes, columns=['x1', 'y1', 'x2', 'y2'])
#         gt_boxes_df['cls'] = gt_labels
#         gt_boxes_df['seg_idx'] = seg_idx
#     return gt_boxes_df



