import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from tqdm import tqdm

from ..utils.path_utils import make_folder

from ..evaluations.sign_evaluation_gt import BBoxAnnotations, prepare_segment_gt, collect_gt_crops
from ..evaluations.sign_evaluation_prep import (prepare_ssd_outputs_for_eval, prepare_ssd_gt_for_eval,
                                                convert_detections_for_eval, convert_to_all_boxes)
from ..evaluations.sign_evaluation import eval_detector, eval_detector_on_collection, get_type_val_frac, evaluate_on_gt
from ..evaluations.sign_evaluator import SignEvalBasic, SignEvalFast

from ..utils.bbox_utils import convert_bbox_global2local, box_iou
from ..utils.nms import nms

from ..detection.detection_helpers import get_detection_bboxes, collect_detection_crops, plot_crop_list
from ..detection.detection_helpers import convert_detections_to_array, vis_detections
from ..detection.run_gen_ssd_detection import get_detections

from ..datasets.cunei_dataset_segments import CuneiformSegments, get_segment_meta

from ..models.trained_model_loader import get_fpn_ssd_net

from ..visualizations.sign_visuals import plot_tpfp_boxes, visualize_TP_FP


def gen_fptp_visuals(relative_path, ds_version, src_used, detection_src, sign_model_version, classes_to_show,
                     lbl2lbl, num_classes, cunei_mzl_df, font_prop, apply_nms=False, nms_th=0.3,
                     filter_completeness=False, compl_thresh=-1, ncompl_thresh=-1,
                     visualize_TPFP=True, vis_score_th=0.3, store_tfps_figures=True, plot_TPFP_signs=True,
                     keep_only_placed=False, use_custom_list=False,
                     eval_tp_thresh=0.5, eval_bg_thresh=0.2, exp_name=None, single_column=False):

    ## Create figure folder for storage
    if exp_name is None:
        coll_model_path = './figures/{}/{}/{}/'.format(src_used, sign_model_version, ds_version)
    else:
        coll_model_path = './figures/{}/{}/{}/'.format(src_used, exp_name, ds_version)

    make_folder(coll_model_path)


    ## Load dataset
    # config dataset
    only_annotated = True
    only_assigned = False
    if src_used == detection_src[2]:
        only_assigned = True

    # load segments dataset
    dataset = CuneiformSegments(collections=ds_version, relative_path=relative_path,
                                only_annotated=only_annotated, only_assigned=only_assigned,
                                preload_segments=False, use_gray_scale=False)

    ## Load generated annotations
    if src_used == detection_src[2]:
        list_anno_dfs = []
        for ds_ver in ds_version:
            # read annotation file
            column_names = ['imageName', 'folder', 'image_path', 'label', 'newLabel', 'x1', 'y1', 'x2', 'y2', 'width',
                            'height', 'seg_idx',
                            'line_idx', 'pos_idx', 'det_score', 'm_score', 'align_ratio', 'nms_keep', 'compl', 'ncompl']
            annotation_file = '{}results/results_ssd/{}/line_generated_bboxes_refined80_{}.csv'.format(relative_path,
                                                                                                       sign_model_version,
                                                                                                       ds_ver)
            anno_df = pd.read_csv(annotation_file, engine='python', names=column_names)
            list_anno_dfs.append(anno_df)
        anno_df = pd.concat(list_anno_dfs)

        anno_df['bbox'] = np.vstack([np.rint(anno_df.x1.values), np.rint(anno_df.y1.values),
                                     np.rint(anno_df.x2.values), np.rint(anno_df.y2.values)]).transpose().astype(
            int).tolist()
        # only use classes in range
        anno_df = anno_df[anno_df.newLabel < num_classes]

        # IMPORTANT: fill nan values in a way that avoids filtering
        anno_df.nms_keep = anno_df.nms_keep.fillna(1).astype(bool)
        anno_df.compl = anno_df.compl.fillna(50)
        anno_df.ncompl = anno_df.ncompl.fillna(100)

        # keep only imputed/placed signs
        if keep_only_placed:
            anno_df = anno_df[(anno_df.m_score == 1) & (anno_df.det_score == 1)]

        # print some stats
        print(len(anno_df))
        print(anno_df.newLabel.value_counts().describe())


    ## Load detector
    if src_used is 'detection':
        arch_type = 'mobile'  # resnet, mobile
        arch_opt = 1
        width_mult = 0.625  # 0.5 0.625 0.75

        with_64 = False
        with_4_aspects = False
        create_bg_class = False

        num_classes = 240

        # detection filter - right after detector
        test_min_score_thresh = 0.05  # 0.05
        test_nms_thresh = 0.3  # 0.3

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        ### Load Model
        fpnssd_net = get_fpn_ssd_net(sign_model_version, device, arch_type, with_64, arch_opt, width_mult,
                                     relative_path, num_classes, num_c=1)

    ## Load evaluation objects
    eval_basic = SignEvalBasic(sign_model_version, ds_version, eval_tp_thresh)
    eval_fast = SignEvalFast(sign_model_version, ds_version, tp_thresh=eval_tp_thresh, bg_thresh=eval_bg_thresh,
                             num_classes=num_classes)


    ## Customize dataset
    didx_list = range(len(dataset))
    didx_list = didx_list[:30]  # :19  # :35  # :30 for saa01(only here)

    if use_custom_list:
        # use this do generate only images for nice tablets (is faster & saves space)
        if 'testEXT' in ds_version:
            didx_list = [5, 7, 11, 15]  # testEXT
        elif 'saa08' in ds_version:
            didx_list = [5, 8, 17, 23]  # saa08
        elif 'saa10' in ds_version:
            didx_list = [12, 13]  # saa10
        elif 'saa03' in ds_version:
            didx_list = [20, 21, 25, 26]  # saa03
        elif 'saa06' in ds_version:
            didx_list = [8, 15, 24]
        elif 'saa04' in ds_version:
            didx_list = [13, 16, 26, 42, 49, 54]


    ## Iterate over dataset
    list_sign_detections = []
    # for seg_im, seg_idx in dataset:
    for didx in tqdm(didx_list):
        # print(didx)
        seg_im, gt_boxes, gt_labels = dataset[didx]
        # convert gt to numpy format
        gt_boxes, gt_labels = prepare_ssd_gt_for_eval(gt_boxes, gt_labels)

        # access meta
        seg_rec = dataset.get_seg_rec(didx)
        seg_idx = seg_rec.segm_idx
        image_name, scale, seg_bbox, _, view_desc = get_segment_meta(seg_rec)
        print(didx, image_name, view_desc)

        # scale image
        input_im = np.asarray(seg_im)

        # select boxes for plots
        if src_used in detection_src[:2]:
            ## 1) GET DETECTIONS
            if src_used is 'detection':
                # create detections in place
                all_boxes = get_detections(fpnssd_net, device, seg_im, with_64, with_4_aspects,
                                           create_bg_class, test_nms_thresh, test_min_score_thresh)
            else:
                # load saved detections
                # boxes file
                res_name = "{}{}".format(image_name, view_desc)
                if src_used is 'detection_pp':
                    res_path = "{}results/results_ssd/{}_pp/{}".format(relative_path, sign_model_version, ds_version)
                else:
                    res_path = "{}results/results_ssd/{}/{}".format(relative_path, sign_model_version, ds_version)
                boxes_file = "{}/{}_all_boxes.npy".format(res_path, res_name)

                # load detections
                all_boxes = np.load(boxes_file)

        # select boxes for plots
        if src_used is 'alignment':
            ## 2) GET ALIGNMENTS
            # select generated annos per segment
            seg_gen_annos = anno_df[(anno_df.seg_idx == seg_idx) & (anno_df.imageName == image_name)]

            # filter using nms
            if filter_completeness:
                seg_gen_annos = seg_gen_annos[seg_gen_annos.nms_keep]
                if compl_thresh > -1:
                    # filter using compl
                    seg_gen_annos = seg_gen_annos[seg_gen_annos.compl > compl_thresh]
                if ncompl_thresh > -1:
                    # filter using compl
                    seg_gen_annos = seg_gen_annos[seg_gen_annos.ncompl > ncompl_thresh]

            # convert to segment local coordinates
            relative_bboxes = seg_gen_annos.bbox.apply(lambda x: convert_bbox_global2local(x, list(seg_bbox)))

            ransac_list = convert_to_all_boxes(seg_gen_annos, relative_bboxes, scale, 240)
            all_boxes_alignment = [[el] for el in ransac_list]
            # convert to numpy array
            all_boxes = np.array(all_boxes_alignment)

        # convert all_boxes to ndarray (N x 9)
        # [ID, cx, cy, score, x1, y1, x2, y2, idx]  bbox = [4:8]  ctr  = [1:3]
        sign_detections = convert_detections_to_array(all_boxes)

        # [OPTIONAL] filter detections with nms
        if apply_nms:
            if len(sign_detections) > 0:
                ft_boxes = sign_detections[:, 4:8]
                ft_scores = sign_detections[:, 3]
                keep = nms(ft_boxes, ft_scores, threshold=nms_th)
                nms_dets = sign_detections[keep]
                # convert sign_detections to all_boxes
                all_boxes = convert_detections_for_eval([nms_dets[:, 4:8]], [nms_dets[:, 0]], [nms_dets[:, 3]])
                all_boxes = [[el] for el in all_boxes]

                # convert all_boxes AGAIN (because index changed after nms!)
                sign_detections = convert_detections_to_array(all_boxes)

        # EVALUATE generated annos using gt annotations
        list_sign_detections.append(sign_detections)

        # check if annotations for collection available
        # only run if gt available
        if len(gt_boxes) > 0:
            # plot gt bbox annotations
            if False:
                # gt for all classes
                fig, axes = plt.subplots(1, 1, figsize=(25, 15))
                vis_detections(input_im, gt_boxes, scores=None, labels=gt_labels,
                               thresh=0.3, max_vis=500, ax=axes)
                plt.show()

            # evaluate
            if False:
                # standard mAP eval
                eval_basic.eval_segment(all_boxes, gt_boxes, gt_labels, seg_idx)

            if True:
                # fast evaluation
                eval_fast.eval_segment(all_boxes, gt_boxes, gt_labels, seg_idx)

            if True:
                ## TP & FP on tablet

                # get current eval_df
                assert len(eval_fast.list_eval_df) > 0
                cur_eval_df = eval_fast.list_eval_df[-1]

                if visualize_TPFP:
                    # show TP & FP detections on tablet
                    if len(cur_eval_df) > 0:
                        plot_tpfp_boxes(input_im, cur_eval_df, sign_detections, score_th=vis_score_th)
                        # nlargest=len(gt_boxes))
                        if store_tfps_figures:
                            # how to remove border:
                            # https://stackoverflow.com/questions/11837979/removing-white-space-around-a-saved-image-in-matplotlib/27227718

                            plt.gca().set_axis_off()
                            # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                            # plt.margins(0, 0)
                            plt.gca().xaxis.set_major_locator(plt.NullLocator())
                            plt.gca().yaxis.set_major_locator(plt.NullLocator())
                            plt.savefig('{}{}{}_{}_tfps_{}.png'.format(coll_model_path, image_name, view_desc,
                                                                       src_used, sign_model_version), dpi=120,
                                        bbox_inches='tight', pad_inches=0)  # dpi=200  transparent=True
                        plt.show()

                if False:
                    # show detections of specific type on tablet
                    select_tp = (cur_eval_df.det_type == 3) & (cur_eval_df.max_det)
                    select_fp = ~select_tp  # & cur_eval_df.pred.isin(eval_fast.list_gt_boxes_df[-1].cls.unique())
                    select_loc = (cur_eval_df.det_type == 0)
                    select_sim = (cur_eval_df.det_type == 1)
                    select_bg = (cur_eval_df.det_type == 2)

                    # select boxes according to detection idx (that corresponds index in sign_detections)
                    select_indices = cur_eval_df[select_sim].det_idx.astype(int)
                    dets_array = sign_detections[select_indices, :]
                    # show specific detections on tablet
                    vis_detections(input_im, dets_array[:, 4:8], scores=dets_array[:, 3], labels=dets_array[:, 0],
                                   thresh=vis_score_th, max_vis=100, figs_sz=(16, 10))
                    plt.show()

    ## Evaluate complete dataset
    ## AP computation across all samples (taking double detections into account)
    eval_fast.prepare_eval_collection()
    if True:
        df_stats, global_ap = eval_fast.eval_collection()

    tp_num = df_stats.tp.sum()
    fp_num = df_stats.fp.sum()
    p_num = df_stats.num_gt.sum()
    # compute precision and recall
    rec = tp_num / float(p_num)
    prec = tp_num / float(tp_num + fp_num)
    print("prec: {:.3} | rec: {:.3}".format(prec, rec))


    ## Visualize individual detection

    # select classes to show
    # classes_to_show = [30]
    #classes_to_show = [108, 96, 30]
    #classes_to_show = [10, 15, 13]

    # define figure path template
    tpfp_figure_path = '{}{}{}_{}_class_{}_{}_{}.png'.format(coll_model_path, image_name, view_desc,
                                                             src_used, "{}", "{}", sign_model_version)

    # fix for custom didx_list (be more flexible then plotting)
    customdidx2didx = dict(zip(np.array(didx_list), np.arange(len(didx_list))))

    col_eval_df = eval_fast.col_eval_df

    ## Access ground truth boxes for each detection
    gt_boxes_df = eval_fast.gt_boxes_df.copy()
    # create gt_idx column that is relative to each seg_idx (see prepare_eval_df() in sign_evaluation)
    gt_boxes_df['gt_idx'] = gt_boxes_df.groupby('seg_idx').cumcount()
    # create multi index for easy access
    gt_boxes_df.index = pd.MultiIndex.from_arrays(gt_boxes_df[['seg_idx', 'gt_idx']].values.T, names=['idx1', 'idx2'])

    if plot_TPFP_signs:
        visualize_TP_FP(classes_to_show, list_sign_detections, dataset, col_eval_df, gt_boxes_df, customdidx2didx,
                        lbl2lbl, cunei_mzl_df, font_prop, tpfp_figure_path, context_pad=40, single_column=single_column,
                        store_figures=store_tfps_figures)



