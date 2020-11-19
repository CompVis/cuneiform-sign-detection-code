import numpy as np
import pandas as pd

from tqdm import tqdm

from .config import cfg

from ..detection.detection_helpers import convert_detections_to_array
from ..utils.bbox_utils import box_iou


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).

    Reference: Ross Girshick's Fast/er R-CNN code
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


# *BASIC* AP COMPUTATION (Fast RCNN style)

def evaluate_on_gt(gt_boxes, gt_labels, num_images, all_boxes, ovthresh=None, num_classes=None, use_07_metric=False):
    # Reference: Ross Girshick's Fast/er R-CNN code

    if ovthresh is None:
        ovthresh = cfg.TEST.TP_MIN_OVERLAP
    if num_classes is None:
        num_classes = cfg.TEST.NUM_CLASSES

    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_tp = [[[] for _ in xrange(num_images)]
                for _ in xrange(num_classes)]
    all_fp = [[[] for _ in xrange(num_images)]
                for _ in xrange(num_classes)]
    det_stats = []
    total_num_tp = 0
    total_false_cls = np.zeros(num_classes)
    for j in xrange(1, num_classes):  # num_classes
        # if no detections for class available
        if len(all_boxes[j][0]) == 0:
            BB = np.empty((0, 4), dtype=np.float32)
            confidence = np.empty(0, dtype=np.float32)
        else:
            BB = all_boxes[j][0][:, :4]
            confidence = all_boxes[j][0][:, -1]

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        inds = np.where(gt_labels == j)[0]
        BBGT = gt_boxes[inds, :].astype(float)
        npos = BBGT.shape[0]
        det = [False] * npos

        if npos > 0:  # else if no gt boxes available for class, AP computation is not meaningful

            # go down dets and mark TPs and FPs
            nd = len(sorted_ind)
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            cls_tp = []
            cls_fp = []
            for d in range(nd):
                bb = BB[d, :].astype(float)
                ovmax = -np.inf

                if BBGT.size > 0:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(BBGT[:, 0], bb[0])
                    iymin = np.maximum(BBGT[:, 1], bb[1])
                    ixmax = np.minimum(BBGT[:, 2], bb[2])
                    iymax = np.minimum(BBGT[:, 3], bb[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih

                    # union
                    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                           (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                           (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)

                if ovmax > ovthresh:
                    if not det[jmax]:
                        tp[d] = 1.
                        det[jmax] = 1
                        cls_tp.append(d)
                    else:
                        # double detection (unlikely due to nms)
                        fp[d] = 1.
                        cls_fp.append(d)  # comment?!
                else:
                    fp[d] = 1.
                    cls_fp.append(d)

            # save tp detections
            all_tp[j][0] = np.array(cls_tp)
            # save fp detections
            all_fp[j][0] = np.array(cls_fp)
            # compute precision recall
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / float(npos)
            # avoid divide by zero in case the first detection matches a difficult
            # ground truth
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = voc_ap(rec, prec, use_07_metric)
            # print rec, prec, ap
            num_tp = np.sum(det).astype(int)
            total_num_tp += num_tp
            det_stats.append([npos, nd, num_tp, nd-num_tp, ap, j])
        else:
            if len(BB) > 0:
                total_false_cls[j] += len(BB)
                #print 'outlier class:', j, len(BB)
    select_nonzero = total_false_cls > 0
    # print(np.nonzero(select_nonzero), total_false_cls[select_nonzero])
    return all_tp, all_fp, det_stats, total_num_tp  #, total_false_cls


def df_evaluate_on_gt(gt_boxes_df, pred_boxes_df, ovthresh=None, num_classes=None, use_07_metric=False):
    # Reference: Ross Girshick's Fast/er R-CNN code

    if ovthresh is None:
        ovthresh = cfg.TEST.TP_MIN_OVERLAP
    if num_classes is None:
        num_classes = cfg.TEST.NUM_CLASSES
    num_images = gt_boxes_df.seg_idx.nunique()

    # sort by confidence
    pred_boxes_df = pred_boxes_df.sort_values('conf', ascending=False)

    det = [False] * len(gt_boxes_df)

    det_stats = []
    total_num_tp = 0
    for j in tqdm(xrange(1, num_classes)):  # num_classes
        cls_dets_df = pred_boxes_df[pred_boxes_df.cls == j]
        cls_gt_df = gt_boxes_df[gt_boxes_df.cls == j]

        # get bounding box and image ids
        BB = cls_dets_df[['x1', 'y1', 'x2', 'y2']].values
        image_ids = cls_dets_df.seg_idx.values
        # confidence = cls_dets_df.conf.values

        npos = len(cls_gt_df)

        if npos > 0:  # else if no gt boxes available for class, AP computation is not meaningful

            # go down dets and mark TPs and FPs
            nd = len(cls_dets_df)
            tp = np.zeros(nd)
            fp = np.zeros(nd)

            for d in range(nd):
                ovmax = -np.inf

                # get bbox and seg_idx
                bb = BB[d, :].astype(float)
                seg_idx = image_ids[d]

                # get gt boxes
                seg_cls_gt_df = cls_gt_df[cls_gt_df.seg_idx == seg_idx]
                BBGT = seg_cls_gt_df[['x1', 'y1', 'x2', 'y2']].values.astype(float)

                if BBGT.size > 0:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(BBGT[:, 0], bb[0])
                    iymin = np.maximum(BBGT[:, 1], bb[1])
                    ixmax = np.minimum(BBGT[:, 2], bb[2])
                    iymax = np.minimum(BBGT[:, 3], bb[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih

                    # union
                    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                           (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                           (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)

                if ovmax > ovthresh:
                    # map seg_cls idx to global idx
                    gidx = seg_cls_gt_df.index.values[jmax]
                    if not det[gidx]:
                        tp[d] = 1.
                        det[gidx] = 1
                    else:
                        # double detection (unlikely due to nms)
                        fp[d] = 1.
                else:
                    fp[d] = 1.
            # compute num tp before cumsum (!)
            num_tp = np.sum(tp).astype(int)
            # compute precision recall
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / float(npos)
            # avoid divide by zero in case the first detection matches a difficult
            # ground truth
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = voc_ap(rec, prec, use_07_metric)
            # print rec, prec, ap
            total_num_tp += num_tp
            det_stats.append([npos, nd, num_tp, nd-num_tp, ap, j])
            # print np.sum(det), total_num_tp
        else:
            if len(cls_dets_df) > 0:
                if False:  # turn on for debugging to see which classes are missing
                    print('outlier class:', j, len(BB))

    return det_stats, total_num_tp


def eval_detector(gt_boxes, gt_labels, all_boxes, ovthresh=None, verbose=True):
    # evaluate
    num_imgs = 1
    all_tp, all_fp, det_stats, total_num_tp = evaluate_on_gt(gt_boxes, gt_labels, num_imgs, all_boxes,
                                                             ovthresh=ovthresh)

    total_num_fp = int(np.sum(np.array(det_stats)[:, 3]))
    # print stats
    pd.set_option('display.max_rows', 50)
    df_stats = pd.DataFrame(det_stats, columns=['num_gt', 'num_det', 'tp', 'fp', 'ap', 'lbl'])

    if verbose:
        print("total_tp", total_num_tp, "total_fp", total_num_fp,
              "mAP", '{:0.4f}'.format(df_stats['ap'].mean()),
              "mAP(nonzero)", '{:0.4f}'.format(df_stats['ap'].iloc[df_stats['ap'].nonzero()[0]].mean()))
    acc = total_num_tp / float(total_num_tp + total_num_fp)

    return acc, df_stats


def eval_detector_on_collection(gt_boxes_df, pred_boxes_df, ovthresh=None):
    det_stats, total_num_tp = df_evaluate_on_gt(gt_boxes_df, pred_boxes_df, ovthresh=ovthresh)

    total_num_fp = int(np.sum(np.array(det_stats)[:, 3]))
    # print stats
    pd.set_option('display.max_rows', 50)
    df_stats = pd.DataFrame(det_stats, columns=['num_gt', 'num_det', 'tp', 'fp', 'ap', 'lbl'])

    print('RESULTS ON FULL COLLECTION :')
    print("total_tp", total_num_tp, "total_fp", total_num_fp,
          "acc", '{:0.3f}'.format(total_num_tp / float(total_num_tp + total_num_fp)),
          "mAP", '{:0.4f}'.format(df_stats['ap'].mean()),
          "mAP(nonzero)", '{:0.4f}'.format(df_stats['ap'].iloc[df_stats['ap'].nonzero()[0]].mean()))
    acc = total_num_tp / float(total_num_tp + total_num_fp)

    return acc, df_stats


# *FAST* AP COMPUTATION


# prepare AP computation


def add_max_det(group):
    # add column to dataframe
    group['max_det'] = False
    # select detections marked as TP
    tp_group = group[group.det_type == 3]
    # only one can be TP, others are double detections
    if len(tp_group) > 0:
        # set max entry to true
        group.max_det.loc[tp_group.score.idxmax()] = True
    return group


def add_det_type_column(eval_df, tp_thresh=0.5, bg_thresh=0.2):
    # based on "Diagnosing Error in Object Detectors" by Hoiem et al.
    # modifications:
    # sim and other categories are merged, since every sign is considered similar
    # bg_thresh is 0.2 instead of default 0.1

    # determine detection types

    type_list = []
    for didx, det_rec in eval_df.iterrows():
        overlap = det_rec.overlap
        # class matches
        if det_rec.pred == det_rec.true:
            if overlap > tp_thresh:
                type_list.append(3)  # TP (3)
            elif overlap > bg_thresh:
                type_list.append(0)  # FP: Loc(0) confusion
            else:
                type_list.append(2)  # FP: BG(2) confusion
        else:
            if overlap > bg_thresh:
                type_list.append(1)  # FP: Sim/Oth(1) confusion
            else:
                type_list.append(2)  # FP: BG(2) confusion

    # add column to dataframe
    eval_df['det_type'] = type_list

    return eval_df


def prepare_eval_df(all_boxes, gt_boxes, gt_labels, seg_idx, tp_thresh, bg_thresh):
    """ prepare eval_df that contains most information for average precision computation """
    # convert all_boxes to ndarray (N x 9)
    # [ID, cx, cy, score, x1, y1, x2, y2, idx]  bbox = [4:8]  ctr  = [1:3]
    sign_detections = convert_detections_to_array(all_boxes)

    # compute ious between detections and gt_boxes
    ious = box_iou(sign_detections[:, 4:8], gt_boxes)

    # for each detection get best fit with gt box
    index_gt = np.argmax(ious, axis=1)
    overlap_gt = np.max(ious, axis=1)
    label_gt = gt_labels[index_gt]

    # collect in data frame
    eval_df = pd.DataFrame(np.hstack([overlap_gt.reshape(-1, 1), label_gt.reshape(-1, 1),
                                      sign_detections[:, [0, 3, 8]], index_gt.reshape(-1, 1)]),
                           columns=['overlap', 'true', 'pred', 'score', 'det_idx', 'gt_idx'])
    # add column with segment index
    eval_df['seg_idx'] = seg_idx
    # add det_type column (0:LOC, 1:SIM, 2:BG, 3:TP)
    eval_df = add_det_type_column(eval_df, tp_thresh, bg_thresh)
    # compute max_det (in order to fin double detections)
    eval_df = eval_df.groupby('gt_idx').apply(add_max_det)

    return eval_df


# AP computation


def compute_mean_ap(col_eval_df, gt_df, num_classes=240, class_list=None, verbose=True):
    """ compute mean class AP """

    # define list of classes to evaluate over
    if class_list is None:
        class_list = np.arange(1, num_classes)  # range(1, num_classes)
    col_eval_df = col_eval_df.sort_values('score', ascending=False)
    if False:
        # filter gt according to considered segments
        bbox_anno = None
        gt_df = bbox_anno.anno_df[bbox_anno.anno_df.segm_idx.isin(col_eval_df.seg_idx.unique())]
        gt_df['cls'] = gt_df.train_label

    # compute class counts
    gt_counts = gt_df.cls.value_counts()

    det_stats = []
    for cls_idx in class_list:
        # get class predictions
        cls_det_df = col_eval_df[col_eval_df.pred == cls_idx]
        # get gt number
        if cls_idx in gt_counts.index:
            npos = gt_counts[cls_idx]
        else:
            npos = 0
        if npos > 0:
            if 1:
                tp_vec = (cls_det_df.det_type == 3) & (cls_det_df.max_det == True)
                fp_vec = ~tp_vec
                # fp_vec = (cls_det_df.det_type < 3) | (cls_det_df.max_det == False)
                fp = np.cumsum(fp_vec.values)
                tp = np.cumsum(tp_vec.values)

                assert np.all(tp_vec != fp_vec), np.intersect1d(tp_vec, fp_vec)
            else:
                # without considering double detections
                fp = np.cumsum(cls_det_df.det_type < 3)
                tp = np.cumsum(cls_det_df.det_type == 3)

            rec = tp / float(npos)
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = voc_ap(rec, prec, False)
            # sum is used to map empty list to 0
            det_stats.append([npos, len(cls_det_df), np.sum(tp[-1:]), np.sum(fp[-1:]), ap, cls_idx])
        else:
            if len(cls_det_df) > 0:
                if False:  # turn on for debugging to see which classes are missing
                    print('outlier class:', cls_idx, len(cls_det_df))
    # convert to ndarray
    det_stats = np.asarray(det_stats)
    mean_ap = np.mean(det_stats[:, -2])
    # return aps
    if verbose:
        print('mAP {:.4}'.format(mean_ap))
    return det_stats


def compute_global_ap(col_eval_df, gt_df, num_classes=240, verbose=True):
    """ compute global AP """

    # sort according to score
    col_eval_df = col_eval_df.sort_values('score', ascending=False)
    # not necessary, because predict classes are only in range [1, num_classes] anyways
    cls_det_df = col_eval_df[col_eval_df.pred.isin(range(1, num_classes))]
    if False:
        # filter gt according to considered segments
        bbox_anno = None
        gt_df = bbox_anno.anno_df[bbox_anno.anno_df.segm_idx.isin(col_eval_df.seg_idx.unique())]
        gt_df['cls'] = gt_df.train_label
    # filter considered classes
    gt_df = gt_df[gt_df.cls.isin(range(1, num_classes))]

    # select number of gt positives
    npos = len(gt_df)
    # npos = len(bbox_anno.anno_df.train_label[bbox_anno.anno_df.train_label > 0])

    ap = 0
    if npos > 0:
        if 1:
            tp_vec = (cls_det_df.det_type == 3) & (cls_det_df.max_det == True)
            fp_vec = ~tp_vec
            fp = np.cumsum(fp_vec)
            tp = np.cumsum(tp_vec)

            assert np.all(tp_vec != fp_vec), np.intersect1d(tp_vec, fp_vec)
        else:
            # without considering double detections
            fp = np.cumsum(cls_det_df.det_type < 3)
            tp = np.cumsum(cls_det_df.det_type == 3)

        rec = tp / float(npos)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, False)

        if False:
            from sklearn.metrics import precision_recall_curve, auc
            import matplotlib.pyplot as plt

            # compute normalized PR curve
            precision, recall, _ = precision_recall_curve(tp_vec, cls_det_df.score.values)
            # plot pr curve
            plt.figure()
            plt.step(recall, precision, color='b', alpha=0.2, where='post')
            # plt.step(rec, prec, color='b', alpha=0.2)  # works, but rec values not normalized to [0, 1] range

            # compare different ways to compute VOC AP (ie. area under the precision recall curve)
            # first two methods should produce same results, but there are slight differences
            # in doubt use original VOC AP code
            # https://datascience.stackexchange.com/questions/25119/how-to-calculate-map-for-detection-task-for-the-pascal-voc-challenge
            # https://github.com/rafaelpadilla/Object-Detection-Metrics
            plt.title('voc ap: {:.3} | PR AUC: {:.3} | norm. PR AUC: {:.3}'.format(voc_ap(rec, prec, False),
                                                                                   auc(rec, prec),
                                                                                   auc(recall, precision)))
            plt.show()

    # return ap
    if verbose:
        print('global AP {:.4}'.format(ap))
    return ap


# FP categorization


def get_type_val_frac(fp_type_series, type_values=[0, 1, 2, 3], num_fp_thres=[5, 10, 25, 50, 100]):
    # type_values = [0, 1, 2, 3]
    # num_fp_thres = [5, 10, 25, 50, 100]

    type_val_frac = np.zeros((len(num_fp_thres), len(type_values)))
    for i, thres in enumerate(num_fp_thres):
        type_counts = fp_type_series[:thres].value_counts(normalize=True, sort=True)
        for j, val in enumerate(type_values):
            val_check = type_counts.index.values == val
            if np.any(val_check):
                val_idx = np.argmax(val_check)
                type_val_frac[i, j] = type_counts.iloc[val_idx]
    return type_val_frac




