import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .sign_evaluation_prep import (get_pred_boxes_df, get_gt_boxes_df)
from .sign_evaluation import (eval_detector, eval_detector_on_collection, prepare_eval_df,
                              compute_global_ap, compute_mean_ap)


# *BASIC* EVALUATION

class SignEvalBasic(object):
    def __init__(self, model_version, collection_name, eval_ovthresh=0.5):

        self.model_version = model_version
        self.coll_name = collection_name

        self.eval_ovthresh = eval_ovthresh

        self.list_seg_mean_ap = []
        self.list_df_stats = []
        #self.list_seg_name_with_anno = []

        self.list_pred_boxes_df = []
        self.list_gt_boxes_df = []

        self.gt_boxes_df = pd.DataFrame()
        self.pred_boxes_df = pd.DataFrame()

    def eval_segment(self, all_boxes, gt_boxes, gt_labels, seg_idx, verbose=True):
        # evaluate and print stats
        acc, df_stats = eval_detector(gt_boxes, gt_labels, all_boxes, ovthresh=self.eval_ovthresh, verbose=verbose)

        # collect results
        self.list_seg_mean_ap.append(df_stats['ap'].mean())
        self.list_df_stats.append(df_stats)
        #list_seg_name_with_anno.append(image_name + view_desc)

        # prepare full collection evaluation
        if type(all_boxes[0]) is np.ndarray:
            self.list_pred_boxes_df.append(get_pred_boxes_df([el.tolist() for el in all_boxes], seg_idx))
            self.list_gt_boxes_df.append(get_gt_boxes_df(gt_boxes, gt_labels, seg_idx))
        else:
            self.list_pred_boxes_df.append(get_pred_boxes_df(all_boxes, seg_idx))
            self.list_gt_boxes_df.append(get_gt_boxes_df(gt_boxes, gt_labels, seg_idx))

    def prepare_eval_collection(self):
        self.gt_boxes_df = pd.concat(self.list_gt_boxes_df, ignore_index=True)
        self.pred_boxes_df = pd.concat(self.list_pred_boxes_df, ignore_index=True)

    def eval_collection(self, verbose=True):
        self.prepare_eval_collection()
        acc, df_stats = eval_detector_on_collection(self.gt_boxes_df, self.pred_boxes_df, ovthresh=self.eval_ovthresh)
        return acc, df_stats


# *FAST* EVALUATION

class SignEvalFast(object):
    def __init__(self, model_version, collection_name, tp_thresh=0.5, bg_thresh=0.2, num_classes=240):

        self.model_version = model_version
        self.coll_name = collection_name

        self.tp_thresh = tp_thresh
        self.bg_thresh = bg_thresh
        self.num_classes = num_classes

        self.list_seg_mean_ap = []

        self.list_eval_df = []
        self.list_gt_boxes_df = []
        self.list_seg_global_ap = []

        self.col_eval_df = pd.DataFrame()
        self.gt_boxes_df = pd.DataFrame()

    def eval_segment(self, all_boxes, gt_boxes, gt_labels, seg_idx, verbose=True):
        # get eval_df
        eval_df = prepare_eval_df(all_boxes, gt_boxes, gt_labels, seg_idx, self.tp_thresh, self.bg_thresh)
        # get gt_df
        gt_df = get_gt_boxes_df(gt_boxes, gt_labels, seg_idx)

        mean_ap, global_ap, mean_ap_align = 0., 0., 0.
        if len(eval_df) > 0 and len(gt_df[gt_df.cls > 0]) > 0:
            # eval
            det_stats = compute_mean_ap(eval_df, gt_df, self.num_classes, verbose=False)
            global_ap = compute_global_ap(eval_df, gt_df, self.num_classes, verbose=False)
            df_stats = pd.DataFrame(det_stats, columns=['num_gt', 'num_det', 'tp', 'fp', 'ap', 'lbl'])
            mean_ap = np.mean(df_stats.ap)
            # mean_ap_align = np.mean(df_stats.ap[df_stats.ap.nonzero()[0]])
            mean_ap_align = np.mean(df_stats.ap[df_stats.num_det > 0])  # only consider classes with detections
            if verbose:
                print ('mAP {:.4} | global AP: {:.4} | mAP (align): {:.4}'.format(mean_ap, global_ap, mean_ap_align))
                print ("total_tp: {} | total_fp: {} [{}] | acc: {:.2}".format(*get_summary(eval_df, gt_df)))
        else:
            if verbose:
                print ('mAP {:.4} | global AP: {:.4} | mAP (align): {}'.format(mean_ap, global_ap, mean_ap_align))
                print ("total_tp: {} | total_fp: {} [{}] | acc: {:.2}".format(0, 0, 0, 0.))

        # append
        self.list_seg_mean_ap.append(mean_ap)
        self.list_seg_global_ap.append(global_ap)
        self.list_eval_df.append(eval_df)
        self.list_gt_boxes_df.append(gt_df)

    def prepare_eval_collection(self, verbose=False):
        if len(self.col_eval_df) == 0:
            if len(self.list_eval_df) > 0:  # only concat if there is anything to concat
                # concat dataframes
                self.col_eval_df = pd.concat(self.list_eval_df)
                self.gt_boxes_df = pd.concat(self.list_gt_boxes_df, ignore_index=True)

        if verbose:
            print(self.col_eval_df.det_type.value_counts())
            print("num det:", len(self.col_eval_df))
            print("num TP (without double detections):",
                  len(self.col_eval_df[(self.col_eval_df.max_det == True)
                                       & (self.col_eval_df.det_type == 3)]))

    def eval_collection(self, verbose=True):
        # concat dataframes
        self.prepare_eval_collection()

        global_ap = 0
        df_stats = pd.DataFrame()
        if len(self.gt_boxes_df) > 0:
            # full collection eval
            det_stats = compute_mean_ap(self.col_eval_df, self.gt_boxes_df, self.num_classes, verbose=False)
            global_ap = compute_global_ap(self.col_eval_df, self.gt_boxes_df, self.num_classes, verbose=False)
            df_stats = pd.DataFrame(det_stats, columns=['num_gt', 'num_det', 'tp', 'fp', 'ap', 'lbl'])
            mean_ap = np.mean(det_stats[:, -2])
            mean_ap_align = np.mean(df_stats.ap[df_stats.num_det > 0])  # only consider classes with detections
            if verbose:
                print('{} | {}'.format(self.coll_name, self.model_version))
                print('RESULTS ON FULL COLLECTION :')
                print ('mAP {:.4} | global AP: {:.4} | mAP (align): {:.4}'.format(mean_ap, global_ap, mean_ap_align))
                print ("total_tp: {} | total_fp: {} [{}] | prec: {:.3}".format(*self.get_col_summary()))

        return df_stats, global_ap

    def eval_collection_class_freq(self, freq_classes_list):
        # freq_classes_list: sorted list of most frequent classes (in descending order)

        # concat dataframes
        self.prepare_eval_collection()

        # compute mAP for different sets of topk most frequent classes
        topk_list = [2, 4, 8, 16, 32, 64, 128, 192, 256]
        topk_mAP_list = []
        for topk in topk_list:
            print("over {} most freq classes".format(topk))
            det_stats = compute_mean_ap(self.col_eval_df, self.gt_boxes_df, self.num_classes,
                                       class_list=freq_classes_list[:topk])
            mean_ap = np.mean(det_stats[:, -2])
            topk_mAP_list.append(mean_ap)
        # plot
        plt.figure()
        plt.plot(topk_list, topk_mAP_list, "o-")
        plt.title('{} - {}'.format(self.coll_name, self.model_version))
        plt.ylabel('mAP')
        plt.xlabel('topk')
        # plt.xscale('log')

    def get_seg_summary(self, didx):
        """ didx: index of segment in list of segments to evaluate """
        num_tp, num_fp, num_fp_global, acc = get_summary(self.list_eval_df[didx], self.list_gt_boxes_df[didx])
        mean_ap = self.list_seg_mean_ap
        global_ap = self.list_seg_global_ap
        return num_tp, num_fp, num_fp_global, acc, mean_ap, global_ap

    def get_col_summary(self):
        num_tp, num_fp, num_fp_global, acc = get_summary(self.col_eval_df, self.gt_boxes_df)
        return num_tp, num_fp, num_fp_global, acc


def get_summary(col_eval_df, gt_boxes_df):
    if len(gt_boxes_df) > 0 and len(col_eval_df) > 0:
        select_tp = (col_eval_df.det_type == 3) & (col_eval_df.max_det == True)
        select_fp = (~select_tp) & col_eval_df.pred.isin(gt_boxes_df.cls.unique())
        num_tp = select_tp.sum()
        num_fp = select_fp.sum()
        num_fp_global = (~select_tp).sum()
        return num_tp, num_fp, num_fp_global, num_tp / float(num_tp + num_fp)
    else:
        return 0, 0, 0, 0.

