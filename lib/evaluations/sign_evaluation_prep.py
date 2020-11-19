import pandas as pd
import numpy as np


def get_pred_boxes_df(all_boxes, seg_idx):
    # iterate list
    list_boxes = []
    list_cls_idx = []
    for cls, boxes in enumerate(all_boxes):
        num_boxes = len(boxes[0])
        if num_boxes > 0:
            list_boxes.append(boxes)
            list_cls_idx.extend([cls] * num_boxes)
    # create df
    pred_boxes_df = pd.DataFrame()  # []
    if len(list_boxes) > 0:
        pred_boxes_df = pd.DataFrame(np.hstack(list_boxes).reshape(-1, 5), columns=['x1', 'y1', 'x2', 'y2', 'conf'])
        pred_boxes_df['cls'] = list_cls_idx
        pred_boxes_df['seg_idx'] = seg_idx

    return pred_boxes_df


def get_gt_boxes_df(gt_boxes, gt_labels, seg_idx):
    # create df
    gt_boxes_df = pd.DataFrame()  # []
    if len(gt_boxes) > 0:
        gt_boxes_df = pd.DataFrame(gt_boxes, columns=['x1', 'y1', 'x2', 'y2'])
        gt_boxes_df['cls'] = gt_labels
        gt_boxes_df['seg_idx'] = seg_idx
    return gt_boxes_df


# SSD specific


def convert_detections_for_eval(pred_boxes, pred_labels, pred_scores, total_labels=240):

    # convert from ssd detector format to all_boxes
    all_boxes = [[] for _ in range(total_labels)]

    for boxes, labels, scores in zip(pred_boxes, pred_labels, pred_scores):
        for bbox, lbl, score in zip(boxes, labels, scores):
            # temp: [ID, cx, cy, score, x1, y1, x2, y2, idx]

            # copy data to _new_ all_boxes
            box = np.zeros((1, 5))
            box[0, :4] = bbox
            box[0, 4] = score
            all_boxes[np.int(lbl)].append(box)

    # for each class stack list of bounding boxes together
    all_boxes = [np.stack(el).squeeze(axis=1) if len(el) > 0 else el for el in all_boxes]

    return all_boxes


def prepare_ssd_outputs_for_eval(box_preds, label_preds, score_preds, num_classes=240):

    if len(box_preds) > 0:
        # Wrap VOC evaluation for PyTorch
        pred_boxes = [b.numpy() for b in [box_preds]]
        pred_labels = [label.numpy() for label in [label_preds]]
        pred_scores = [score.numpy() for score in [score_preds]]

        # convert to all boxes and stack tiles (better would be to have single tile for whole segment)
        all_boxes = convert_detections_for_eval(pred_boxes, pred_labels, pred_scores, num_classes)
        all_boxes = [[el] for el in all_boxes]
    else:
        # deal with case if there are not any detections
        all_boxes = [[] for _ in range(num_classes)]
        all_boxes = [[el] for el in all_boxes]

    return all_boxes


def prepare_ssd_gt_for_eval(gt_boxes, gt_labels):
    gt_boxes = [b.numpy() for b in [gt_boxes]]
    gt_labels = [label.numpy() for label in [gt_labels]]

    return gt_boxes[0], gt_labels[0]


# alignment specific

def convert_to_all_boxes(seg_gen_annos, relative_bboxes, scale, num_labels):
    all_boxes = [[] for _ in range(num_labels)]

    for anno_idx, anno_rec in seg_gen_annos.iterrows():
        # [x1, y1, x2, y2, score]
        box = np.zeros((1, 5))
        box[0, :4] = np.array(relative_bboxes[anno_idx]) * scale
        box[0, 4] = anno_rec.det_score
        # assign to class
        all_boxes[anno_rec.newLabel].append(box)

    # for each class stack list of bounding boxes together
    all_boxes = [np.stack(el).squeeze(axis=1) if len(el) > 0 else el for el in all_boxes]

    return all_boxes

