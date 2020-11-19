import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from skimage.color import label2rgb

from ..transliteration.TransliterationSet import TransliterationSet
from ..transliteration.SignsStats import SignsStats

from ..evaluations.sign_tl_evaluation import compute_accuracy
from ..evaluations.line_tl_evaluation import eval_line_tl_alignment
from ..evaluations.sign_evaluation_prep import get_pred_boxes_df, get_gt_boxes_df
from ..evaluations.sign_evaluation_gt import prepare_segment_gt
from ..evaluations.sign_evaluation import eval_detector_on_collection
from ..evaluations.sign_evaluator import SignEvalBasic, SignEvalFast

from ..alignment.line_tl_alignment import compute_line_tl_alignment
from ..alignment.LineFragment import (LineFragment, compute_line_points, compute_line_polygon, plot_boxes)

from ..detection.line_detection import (prepare_transliteration, preprocess_line_input, apply_detector,
                                        post_process_line_detections, compute_image_label_map)
from ..detection.detection_helpers import (visualize_net_output, radius_in_image, convert_detections_to_array,
                                                label_map2image, vis_detections, coord_in_image)
#from ..detection.tablet_scale_estimation import print_scale_stats

from ..visualizations.line_visuals import (show_hough_transform_w_lines, show_line_segms, show_line_skeleton, show_probabilistic_hough)
from ..visualizations.line_tl_visuals import show_lines_tl_alignment, show_score_mats_with_paths


def gen_alignments(didx_list, dataset, bbox_anno, lines_anno, relative_path, saa_version, re_transform,
                   sign_model_version, model_fcn, device,
                   generate_and_save, show_sign_alignments, collection_subfolder, train_data_ext_file, lbl_list,
                   line_model_version='v007', use_precomp_lines=False, param_dict=None,
                   show_line_matching=False, verbose=True):
    """
    Generate tl-line pairs for seq model training. Store pairs in file.
    Additionally compute some useful filter criterion for generated pairs.
    """

    # config tl_line matching
    # 1: line length
    # 2: use gt line anno (if available)
    # 3: shortest path through score matrix
    align_opt = [False, False, True]
    visualize_tl_line_matching = show_line_matching

    # setup evaluators
    use_new_eval = True
    num_classes = 240
    eval_ovthresh = 0.5
    eval_basic = SignEvalBasic(sign_model_version, saa_version, eval_ovthresh)
    eval_fast = SignEvalFast(sign_model_version, saa_version, tp_thresh=eval_ovthresh, num_classes=num_classes)

    # setup transliteration set
    tl_set = TransliterationSet(collections=[saa_version], relative_path=relative_path)
    # setup sign statistics
    stats = SignsStats(tblSignHeight=128)

    list_pred_boxes_df, list_gt_boxes_df = [], []
    acc_array = np.zeros(len(didx_list))
    naligned_array = np.zeros(len(didx_list))
    for didx in tqdm(didx_list, desc=saa_version):
        seg_im, seg_idx = dataset[didx]
        # access meta
        seg_rec = dataset.assigned_segments_df.loc[seg_idx]
        image_name, scale, seg_bbox, image_path, view_desc = dataset.get_segment_meta(seg_rec)
        print(didx, image_name, view_desc)

        # load transliteration dataframe
        tl_df, num_lines = tl_set.get_tl_df(seg_rec, verbose=verbose)
        tl_df, num_vis_lines, len_min, len_max = prepare_transliteration(tl_df, num_lines, stats)
        #print(float(len_min) / len_max, num_vis_lines)

        # boxes file
        res_name = "{}{}".format(image_name, view_desc)
        res_path = "{}results/results_ssd/{}/{}".format(relative_path, sign_model_version, saa_version)
        boxes_file = "{}/{}_all_boxes.npy".format(res_path, res_name)

        # load detections
        all_boxes = np.load(boxes_file)
        sign_detections = convert_detections_to_array(all_boxes)

        # load and prepare annotations of segment
        gt_boxes, gt_labels = prepare_segment_gt(seg_idx, scale, bbox_anno,
                                                 with_star_crop=False)  # depends on sign_detections!
        if verbose:
            print('Load annotations: {} gt bboxes found.'.format(len(gt_boxes)))

        # make seg image is large enough for line detector
        if seg_im.size[0] > 224 and seg_im.size[1] > 224:

            if use_precomp_lines:
                # to numpy
                center_im = np.asarray(seg_im)
                # lbl_ind
                line_res_path = "{}results/results_line/{}/{}".format(relative_path, line_model_version, saa_version)
                lines_file = "{}/{}_lbl_ind.npy".format(line_res_path, res_name)
                # lines_file = "{}/{}_skeleton.npy".format(line_res_path, res_name)
                lbl_ind_x = np.load(lines_file).astype(int)
            else:
                # prepare input
                inputs = preprocess_line_input(seg_im, 1, shift=0)
                center_im = re_transform(inputs[4])  # to pil image
                center_im = np.asarray(center_im)  # to numpy
                # apply network
                #print(inputs.shape)
                output = apply_detector(inputs, model_fcn, device)
                # visualize_net_output(center_im, output, cunei_id=1, num_classes=2)
                # plt.show()

                # prepare output
                outprob = np.mean(output, axis=0)
                lbl_ind = np.argmax(outprob, axis=0)

                lbl_ind_x = lbl_ind.copy()
                lbl_ind_x[np.max(outprob, axis=0) < 0.7] = 0  # line detector dependent (VIP) # outprob.squeeze() # this fixes a bug!

                lbl_ind_80 = lbl_ind.copy()
                lbl_ind_80[np.max(outprob, axis=0) < 0.8] = 0  # outprob.squeeze() # this fixes a bug!

            # only continue if there is a positive line detection
            # (avoids unnecessary computation and an error in skimage hough_line_peaks)
            if np.any(lbl_ind_x):

                # for line detection apply postprocessing pipeline
                (line_hypos, line_segs, segm_labels, ls_labels, dist_interline_median, group2line,
                 h, theta, d, skeleton) = post_process_line_detections(lbl_ind_x, num_vis_lines, len_min, len_max, verbose=verbose)

                if len(line_segs) > 0:
                    # compute overlay
                    seg_canvas = compute_image_label_map(segm_labels, center_im.shape)
                    image_label_overlay = label2rgb(seg_canvas, image=center_im)

                # using line annotations: gt_line_idx for hypo_lines
                gt_line_assignment = lines_anno.get_assignment_for_line_hypos(seg_idx, line_hypos.groupby('label').mean())

                if len(gt_line_assignment) > 0:
                    # clean join on line_hypos
                    line_hypos = line_hypos.join(gt_line_assignment.set_index('hypo_line_lbl'), on='label')
                    ## clean join on line_hypos_agg
                    # line_frag.line_hypos_agg.join(gt_line_assignment.set_index('hypo_line_lbl'))

                if len(tl_df) > 0:

                    # abort if obvious transliteration / lines mismatch
                    if np.abs(tl_df.line_idx.nunique() - line_hypos.label.nunique()) > 10:
                        print("CANCEL segment [{}] : Due to obvious transliteration / lines mismatch".format(seg_idx))
                        continue

                    #### line-transliteration alignment problem ####

                    line_hypos, path_pts = compute_line_tl_alignment(line_hypos, tl_df, gt_line_assignment,
                                                                     segm_labels, stats, center_im, sign_detections,
                                                                     visualize=visualize_tl_line_matching,
                                                                     align_opt=align_opt)

                    # FINISH lines-tl alignment

                    # create line fragment (tl_line should be assigned before?!)
                    line_frag = LineFragment(line_hypos, segm_labels, tl_df, stats, center_im, sign_detections)
                    # get assigned tl indices
                    assigned_tl_indices = line_frag.get_assigned_lines_idx()
                    # get assignment space (cartesian product of tl_line_indices and hypo_line_indices)
                    hypo_line_indices, tl_line_indices = line_frag.get_alignment_space()

                    # evaluate line-tl alignment using gt-line annotations; only quality indicator because unreliable
                    if len(gt_line_assignment) > 0 and verbose:
                        eval_line_tl_alignment(line_frag, lines_anno, seg_idx, num_vis_lines)

                # common colormap
                # color = plt.cm.jet(np.linspace(0,1,len(angles)))
                cmap = plt.get_cmap('nipy_spectral')
                color = cmap(np.linspace(0, 1, len(line_hypos)))

                # estimate scale
                if False:
                    if len(tl_df) == 0:
                        # use line detection estimates
                        num_lines = line_hypos.label.nunique()
                        len_max = line_hypos.groupby('label').mean().accum.max() / dist_interline_median

                    # get scales using different approaches
                    # use num_lines for scale estimation (NOT num_vis_lines!)
                    print_scale_stats(seg_rec, scale, lbl_ind_x, lbl_ind_80, num_lines, len_max,
                                      line_hypos, dist_interline_median)

                if False:
                    show_line_skeleton(lbl_ind_x, skeleton)
                    plt.show()

                if False:
                    show_hough_transform_w_lines(lbl_ind_x, center_im, h, theta, d, line_hypos, color)

                if len(line_segs) > 0:
                    if False:
                        show_probabilistic_hough(lbl_ind_x, center_im, line_segs, ls_labels, group2line, color)

                    if False:
                        show_line_segms(image_label_overlay, segm_labels)

                if len(tl_df) > 0:
                    if False:
                        show_lines_tl_alignment(lbl_ind_x, center_im, line_hypos, color)

                    if False:
                        show_score_mats_with_paths(assigned_tl_indices, hypo_line_indices, tl_line_indices, line_frag)

                    if True:

                        if show_sign_alignments:
                            aligned_list, tablet_tl_df = line_frag.tab_visualize_gm_alignments(refined=True)  # refined=True, does not help/hurt
                        else:
                            refined = False
                            if param_dict is not None:
                                if 'refined' in param_dict:
                                    refined = param_dict['refined']
                            aligned_list, tablet_tl_df = line_frag.tab_get_gm_alignments(refined=refined,
                                                                                         param_dict=param_dict)  # refined=True, does not help/hurt
                        if len(gt_boxes) > 0:

                            if use_new_eval:
                                if len(aligned_list) > 0:
                                    all_boxes = [[el] for el in aligned_list]
                                    if False:
                                        # standard mAP eval
                                        eval_basic.eval_segment(all_boxes, gt_boxes, gt_labels, seg_idx, verbose=verbose)
                                    # fast evaluation
                                    eval_fast.eval_segment(all_boxes, gt_boxes, gt_labels, seg_idx, verbose=verbose)
                                    # get segment statistics of current segment [-1]
                                    num_tp, num_fp, _, acc, mean_ap, global_ap = eval_fast.get_seg_summary(-1)
                                    # save acc to array
                                    acc_array[didx_list.index(didx)] = acc
                                    # save naligned to array
                                    naligned_array[didx_list.index(didx)] = num_tp + num_fp
                            else:
                                # prepare full collection evaluation
                                list_pred_boxes_df.append(get_pred_boxes_df([[el] for el in aligned_list], seg_idx))
                                list_gt_boxes_df.append(get_gt_boxes_df(gt_boxes, gt_labels, seg_idx))

                                # get num aligned across all classes
                                naligned = np.sum([len(el) for i, el in enumerate(aligned_list) if i > 0])

                                if verbose and len(aligned_list) > 0:
                                    # [METHOD B]: evaluate mAP and print stats for a single segment
                                    # (these results can strongly differ from collection-wise evaluation)
                                    acc, df_stats = compute_accuracy(gt_boxes, gt_labels, aligned_list, return_stats=True)
                                    # save acc to array
                                    acc_array[didx_list.index(didx)] = acc

                                    ntfpos = df_stats.tp.sum() + df_stats.fp.sum()
                                    # print ntfpos, naligned

                                    # save naligned to array
                                    naligned_array[didx_list.index(didx)] = ntfpos  # naligned

                        if generate_and_save:
                            line_frag.tab_generate_training_data(collection_subfolder, train_data_ext_file,
                                                                 image_name, image_path, scale, seg_idx, seg_bbox,
                                                                 tablet_tl_df, lbl_list, append=True)

            else:
                print('No lines detected for {}[{}] and thus no alignment performed!'.format(image_name, seg_idx))

        else:
            print('segment image of for {}[{}] too small!'.format(image_name, seg_idx))

        # make plots appear
        plt.show()

    # full collection eval
    acc = 0
    df_stats = []
    if use_new_eval:
        eval_fast.prepare_eval_collection()
        df_stats, global_ap = eval_fast.eval_collection(verbose=verbose)
        num_tp, num_fp, num_fp_global, acc = eval_fast.get_col_summary()
    else:
        if len(list_gt_boxes_df) > 0:
            # [METHOD C]: compute mAP across all instances of individual classes
            # (these results can strongly differ from segment-wise evaluation)
            gt_boxes_df = pd.concat(list_gt_boxes_df, ignore_index=True)
            pred_boxes_df = pd.concat(list_pred_boxes_df, ignore_index=True)
            acc, df_stats = eval_detector_on_collection(gt_boxes_df, pred_boxes_df, ovthresh=None)  # set fixed!
    return acc, df_stats    # acc_array, naligned_array




