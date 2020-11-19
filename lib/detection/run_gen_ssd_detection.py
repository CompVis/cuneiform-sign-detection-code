import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from ..datasets.cunei_dataset_segments import CuneiformSegments, get_segment_meta

from ..alignment.LineFragment import plot_boxes
from ..utils.path_utils import make_folder

from ..utils.torchcv.box_coder_retina import RetinaBoxCoder
from ..utils.torchcv.box_coder_fpnssd import FPNSSDBoxCoder
from ..utils.torchcv.box import box_nms
from ..utils.torchcv.evaluations.voc_eval import voc_eval

from ..evaluations.sign_evaluation_prep import (prepare_ssd_outputs_for_eval, prepare_ssd_gt_for_eval,
                                                get_pred_boxes_df, get_gt_boxes_df)
from ..evaluations.sign_evaluation import eval_detector, eval_detector_on_collection
from ..evaluations.sign_evaluator import SignEvalBasic, SignEvalFast


def gen_ssd_detections(didx_list, dataset, saa_version, relative_path,
                       model_version, fpnssd_net, with_64, create_bg_class, device,
                       test_min_score_thresh, test_nms_thresh, eval_ovthresh,
                       save_detections, show_detections, with_4_aspects=False, verbose_mode=True, return_eval=False):

    list_pred_boxes_df, list_gt_boxes_df = [], []
    list_seg_ap, list_seg_name_with_anno = [], []

    # setup evaluators
    use_new_eval = True
    num_classes = 240
    # eval_basic = SignEvalBasic(model_version, saa_version, eval_ovthresh)
    eval_fast = SignEvalFast(model_version, saa_version, tp_thresh=eval_ovthresh, num_classes=num_classes)

    # iterate over segments
    for didx in tqdm(didx_list, desc=saa_version):
        # print(didx)
        seg_im, gt_boxes, gt_labels = dataset[didx]

        # access meta
        seg_rec = dataset.get_seg_rec(didx)
        image_name, scale, seg_bbox, _, view_desc = get_segment_meta(seg_rec)

        # for plots
        input_im = np.asarray(seg_im)

        # prepare box coder
        # box_coder = RetinaBoxCoder()
        box_coder = FPNSSDBoxCoder(input_size=seg_im.size, with_64=with_64, with_4_aspects=with_4_aspects, create_bg_class=create_bg_class)

        # prepare input
        inputs = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.5], std=[1.0])])(seg_im)
        inputs = inputs.unsqueeze(0)

        with torch.no_grad():
            loc_preds, cls_preds = fpnssd_net(inputs.to(device))

            box_preds, label_preds, score_preds = box_coder.decode(
                loc_preds.cpu().data.squeeze(),
                F.softmax(cls_preds.squeeze(), dim=1).cpu().data,
                score_thresh=test_min_score_thresh, nms_thresh=test_nms_thresh)

        if show_detections:
            # plot prediction
            plt.figure(figsize=(10, 10))
            plot_boxes(box_preds, confidence=score_preds)
            plt.imshow(input_im, cmap='gray')
            plt.grid(True, color='w', linestyle=':')
            plt.show()

            # vis_detections(input_im, box_preds, scores=score_preds, labels=label_preds,
            #                thresh=0.01, max_vis=300, figs_sz=(15, 15))  #lbl2lbl[labels]
            # plt.show()

        # convert detections to all boxes format
        all_boxes = prepare_ssd_outputs_for_eval(box_preds, label_preds, score_preds)

        if save_detections:
            res_name = "{}{}".format(image_name, view_desc)
            res_path = "{}results/results_ssd/{}/{}".format(relative_path, model_version, saa_version)

            # check folder
            make_folder(res_path)

            if True:
                # Save detections
                # outfile = "{}/{}.npy".format(res_path, res_name)
                # np.save(outfile, scores)

                # save all_boxes
                outfile = "{}/{}_all_boxes.npy".format(res_path, res_name)
                np.save(outfile, all_boxes)

        if gt_boxes is not None:

            if 0:
                if verbose_mode:
                    # [METHOD A]: evaluate for a single segment (in tensor format)
                    print(voc_eval([box_preds.clone()], [label_preds.clone()], [score_preds.clone()],
                                   [gt_boxes.clone()], [gt_labels.clone()], None,
                                   iou_thresh=eval_ovthresh, use_07_metric=False)['map'])

            # convert gt to numpy format
            gt_boxes, gt_labels = prepare_ssd_gt_for_eval(gt_boxes, gt_labels)

            if use_new_eval:
                list_seg_name_with_anno.append(image_name + view_desc)
                if verbose_mode:
                    print(image_name, view_desc)
                # standard mAP eval
                # eval_basic.eval_segment(all_boxes, gt_boxes, gt_labels, seg_rec.segm_idx, verbose=verbose_mode)
                # fast evaluation
                eval_fast.eval_segment(all_boxes, gt_boxes, gt_labels, seg_rec.segm_idx, verbose=verbose_mode)
            else:
                if verbose_mode:
                    # [METHOD B]: evaluate mAP and print stats for a single segment
                    # (these results can strongly differ from collection-wise evaluation)
                    acc, df_stats = eval_detector(gt_boxes, gt_labels, all_boxes, ovthresh=eval_ovthresh)
                    # collect results
                    list_seg_ap.append(df_stats['ap'].mean())
                    list_seg_name_with_anno.append(image_name + view_desc)

                # prepare full collection evaluation
                list_pred_boxes_df.append(get_pred_boxes_df(all_boxes, seg_rec.segm_idx))
                list_gt_boxes_df.append(get_gt_boxes_df(gt_boxes, gt_labels, seg_rec.segm_idx))

    # full collection eval
    if use_new_eval:
        eval_fast.prepare_eval_collection()
        df_stats, global_ap = eval_fast.eval_collection(verbose=verbose_mode)
        if return_eval:
            return global_ap, df_stats, eval_fast
        else:
            if verbose_mode:
                return eval_fast.list_seg_mean_ap, list_seg_name_with_anno
            else:
                return global_ap, df_stats
    else:
        acc = 0
        df_stats = pd.DataFrame()
        if len(list_gt_boxes_df) > 0:
            # [METHOD C]: compute mAP across all instances of individual classes
            # (these results can strongly differ from segment-wise evaluation)
            gt_boxes_df = pd.concat(list_gt_boxes_df, ignore_index=True)
            pred_boxes_df = pd.concat(list_pred_boxes_df, ignore_index=True)
            acc, df_stats = eval_detector_on_collection(gt_boxes_df, pred_boxes_df, ovthresh=eval_ovthresh)

        if verbose_mode:
            return list_seg_ap, list_seg_name_with_anno
        else:
            return acc, df_stats


def get_detections(fpnssd_net, device, seg_im, with_64, with_4_aspects, create_bg_class,
                   test_nms_thresh, test_min_score_thresh):
    # prepare box coder
    # box_coder = RetinaBoxCoder()
    box_coder = FPNSSDBoxCoder(input_size=seg_im.size, with_64=with_64, with_4_aspects=with_4_aspects,
                               create_bg_class=create_bg_class)

    # prepare input
    inputs = transforms.Compose([transforms.Lambda(lambda x: x.convert('L')),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.5], std=[1.0])])(seg_im)
    inputs = inputs.unsqueeze(0)

    with torch.no_grad():
        loc_preds, cls_preds = fpnssd_net(inputs.to(device))

        box_preds, label_preds, score_preds = box_coder.decode(
            loc_preds.cpu().data.squeeze(),
            F.softmax(cls_preds.squeeze(), dim=1).cpu().data,
            score_thresh=test_min_score_thresh, nms_thresh=test_nms_thresh)

    # convert detections to all boxes format
    all_boxes = prepare_ssd_outputs_for_eval(box_preds, label_preds, score_preds)

    return all_boxes
