import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from ..transliteration.mzl_util import get_unicode_comp
from ..detection.detection_helpers import crop_bboxes_from_im


# Visualization for full segments

def plot_boxes(boxes, confidence=None, ax=None, label='TP'):
    # handle input
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 12))
    if confidence is None:
        confidence = np.ones(boxes.shape[0]) * 0.8  # 0.2
    # plot
    for ii, bbox in enumerate(boxes):
        # shadow
        shadow = plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],  # transform=shadow_transform,
                               edgecolor="black", fill=False, alpha=0.4, linewidth=4.0)  # alpha=0.3
        ax.add_patch(shadow)
        # box
        rectangle = plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], fill=False,
                                  edgecolor=plt.cm.plasma(confidence[ii]), alpha=0.8,
                                  # alpha=cnn_confidence[ii],
                                  linewidth=2.0)  # linewidth=2.0
        ax.add_patch(rectangle)  # edgecolor='red'


def plot_tpfp_boxes(input_im, cur_eval_df, sign_detections, score_th=0.3, nlargest=0):
    select_tp = (cur_eval_df.det_type == 3) & (cur_eval_df.max_det)
    select_fp = ~select_tp
    # setup plot
    fig, ax = plt.subplots(figsize=(15, 12))
    # either select with fixed threshold or n largest scores detection
    if nlargest > 0:
        select_score = cur_eval_df.index.isin(cur_eval_df.nlargest(nlargest, 'score').index)
    else:
        select_score = cur_eval_df.score > score_th
    # plot fp boxes
    dets_array = sign_detections[cur_eval_df[select_score & select_fp].det_idx.astype(int), :]
    plot_boxes(dets_array[:, 4:8], confidence=np.zeros(dets_array.shape[0]), ax=ax, label='FP')
    # plot tp boxes
    dets_array = sign_detections[cur_eval_df[select_score & select_tp].det_idx.astype(int), :]
    plot_boxes(dets_array[:, 4:8], confidence=np.ones(dets_array.shape[0]), ax=ax, label='TP')
    # plot image
    ax.imshow(input_im, cmap='gray')
    if nlargest > 0:
        ax.set_title('TP: {} | FP: {} | {}'.format((select_score & select_tp).sum(),
                                                   (select_score & select_fp).sum(), nlargest), fontsize=14)
    else:
        ax.set_title('TP: {} | FP: {} | conf >= {:.2f}'.format((select_score & select_tp).sum(),
                                                               (select_score & select_fp).sum(), score_th), fontsize=14)
    ax.axis('off')


# Visualization for indvidual signs

def crop_detections(group_eval_df, list_sign_detections, dataset, customdidx2didx, context_pad=20, return_bboxes=False):
    # iterate over group_eval_df entries and collect crops
    #for idx, eval_rec in tqdm(group_eval_df.iterrows(), total=len(group_eval_df)):
    det_crops = []
    det_bboxes = []
    grouped = group_eval_df.groupby('seg_idx')
    for seg_idx, group in grouped:
        # get segment image and sign detections
        input_im = dataset[dataset.sidx2didx[seg_idx]][0]
        #sign_detections = list_sign_detections[dataset.sidx2didx[seg_idx]]
        sign_detections = list_sign_detections[customdidx2didx[dataset.sidx2didx[seg_idx]]]  # fix for custom didx_list
        # select bounding boxes
        bboxes = sign_detections[group.det_idx.astype(int)][:, 4:8]
        # crop boxes and extend to list (do not preserve individual lists)
        det_crops.extend(crop_bboxes_from_im(input_im, bboxes, context_pad=context_pad, is_pil=True))
        det_bboxes.extend(bboxes)

    if return_bboxes:
        return det_crops, det_bboxes
    else:
        return det_crops


def compute_rel_gt_boxes(gt_boxes_df, group_eval_df, group_bboxes, context_pad=30):
    # this compute relative to detection crops the gt boxes

    # Importantly, a list of tuples indexes several complete MultiIndex keys,
    # whereas a tuple of lists refer to several values within a level:
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#advanced-indexing-with-hierarchical-index
    gt_boxes = gt_boxes_df.loc[group_eval_df[['seg_idx', 'gt_idx']].apply(tuple, axis=1).values]
    gt_boxes = gt_boxes[['x1', 'y1', 'x2', 'y2']]
    gt_boxes = gt_boxes.reset_index()

    # prediction boxes
    pred_boxes = pd.DataFrame(np.vstack(group_bboxes), columns=['x1', 'y1', 'x2', 'y2'])
    # sub_pred_boxes.head()

    # compute relative gt boxes
    gt_boxes['rel_x1'] = gt_boxes.x1 - (pred_boxes.x1 - context_pad)
    gt_boxes['rel_x2'] = gt_boxes.x2 - (pred_boxes.x1 - context_pad)
    gt_boxes['rel_y1'] = gt_boxes.y1 - (pred_boxes.y1 - context_pad)
    gt_boxes['rel_y2'] = gt_boxes.y2 - (pred_boxes.y1 - context_pad)
    # width and height
    gt_boxes['width'] = gt_boxes.x2 - gt_boxes.x1
    gt_boxes['height'] = gt_boxes.y2 - gt_boxes.y1
    return gt_boxes


def visualize_group_detections(group_eval_df, group_crops, lbl2lbl, cunei_mzl_df, font_prop,
                               figs_sz=None, num_cols=6, max_vis=18, color_det_type=True, context_pad=30, gt_boxes=None):
    # visualize sign detections
    # limit number crops (only works if sorted after score beforehand)
    if len(group_crops) > max_vis:
        group_crops = group_crops[:max_vis]
        group_eval_df = group_eval_df.iloc[:max_vis]

    nvis = len(group_crops)
    num_rows = (nvis // num_cols + (nvis % num_cols > 0))
    top_vals = group_eval_df.score.values

    # compute fig size
    patch_width = 2.3
    patch_height = 2.3
    plots_height = patch_height * float(num_rows)

    if 1:
        if figs_sz is None:
            figs_sz = (patch_width * num_cols, plots_height)

        # prepare subplots (nvis or nvis + 1)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=figs_sz, squeeze=False,
                                 gridspec_kw={'height_ratios': [1] * num_rows})  # 'width_ratios': [1],

    else:
        fig, axes = plt.subplots(max_vis, num_cols, figsize=(patch_width * num_cols, patch_height * max_vis), squeeze=False,
                                 gridspec_kw={'height_ratios': [1] * max_vis})  # 'width_ratios': [1],


    # , gridspec_kw={'wspace': 1}
    axes = axes.ravel()

    # iterate over top_list
    for i, ax in enumerate(axes):
        if (i < len(group_crops)) and (i < num_rows):
            eval_rec = group_eval_df.iloc[i]
            imcrop = group_crops[i]
            im_handle = ax.imshow(imcrop)  # , cmap=plt.cm.Greys_r
            # ax.set_title("pred:{} | p(x)={:.1f}".format(int(eval_rec.pred), eval_rec.score), fontsize=8)
            # ax.set_title(get_unicode_comp(lbl2lbl[int(eval_rec.pred)], cunei_mzl_df), fontproperties=font_prop, fontsize=10)
            ax.set_yticks([])
            ax.set_xticks([])
            # align on left side
            ax.set_anchor('W')

            if context_pad > 0:
                imw, imh = imcrop.shape[:2]
                bbox = [context_pad, context_pad, imh - context_pad, imw - context_pad]

                # add shadow
                shadow = plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                       edgecolor="black", fill=False, alpha=0.5, linewidth=3.0)  # alpha=0.3
                ax.add_patch(shadow)

                gt_box_color = '#DAA520'  # iceblue/nice: '#a6cee3', gold/good: #DAA520,
                # good: '#cccccc', better: '#fdbf6f', ok: '#fb9a99',
                # add bounding box
                box_color = 'blue'  # good:'#2c7bb6', ok: '#105e98', clear/dark: 'blue'
                det_type_txt = 'TP'
                if color_det_type:
                    if eval_rec.det_type == 0 or (eval_rec.det_type == 3 and not eval_rec.max_det):
                        box_color = '#1f78b4'  # 'yellow' # localization
                        det_type_txt = 'Loc'
                    elif eval_rec.det_type == 1:
                        box_color = '#b2df8a'  # 'cyan'  # class confusion
                        det_type_txt = 'Cls'
                    elif eval_rec.det_type == 2:
                        box_color = '#7b3294'  # 'purple' # background
                        det_type_txt = 'BG'
                rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                     fill=False, edgecolor=box_color, linestyle='-',
                                     alpha=0.8, linewidth=1.5)  # alpha=0.3
                ax.add_patch(rect)

                # add gt bounding box
                if 1:
                    if gt_boxes is not None:
                        if eval_rec.det_type != 2:  # not background
                            gt_bbox_rec = gt_boxes.iloc[i]
                            gt_bbox = gt_bbox_rec[['rel_x1', 'rel_y1', 'width', 'height']].values  # x,y,w,h
                            gt_rect = plt.Rectangle((gt_bbox[0], gt_bbox[1]), gt_bbox[2], gt_bbox[3],
                                                    fill=False, edgecolor=gt_box_color, linestyle='--', alpha=0.6,  # : -. --
                                                    linewidth=2.0)
                            ax.add_patch(gt_rect)

                # print information in corners
                if 0:  # top left corner
                    # show score of predicted class
                    box_text = '{:.2f}'.format(eval_rec.score)  # p(x)=
                    ax.text(bbox[0] / 2, bbox[1], box_text, bbox=dict(facecolor='blue', alpha=0.4, pad=2),
                            fontsize=12, color='white')

                if 0:  # bottom left corner
                    # show det type
                    ax.text(bbox[0] / 2, bbox[3] - bbox[1] / 8, det_type_txt,
                            bbox=dict(facecolor=box_color, alpha=0.4, pad=2),  # 'red'
                            fontsize=13, color='white')

                if 0:  # top right corner
                    # show predicted class
                    box_text = get_unicode_comp(lbl2lbl[int(eval_rec.pred)], cunei_mzl_df)
                    ax.text(bbox[2] - bbox[0] / 2, bbox[1], box_text, bbox=dict(facecolor='blue', alpha=0.4, pad=2),
                            fontsize=12, color='white', fontproperties=font_prop)

                if 0:  # bottom right corner
                    if eval_rec.det_type != 2:  # not background
                        # show true class
                        box_text = get_unicode_comp(lbl2lbl[int(eval_rec.true)], cunei_mzl_df)
                        ax.text(bbox[2] - bbox[0] / 2, bbox[3] - bbox[1] / 8, box_text,
                                bbox=dict(facecolor='red', alpha=0.4, pad=2),
                                fontsize=12, color='white', fontproperties=font_prop)

                # print information below image ("sub title")
                ####ax.set_title("Test", y=-.2, fontsize=12)
                if True:
                    if 0:
                        # show predicted class below
                        ax.text(0.0, -0.11, 'P:', horizontalalignment='left', verticalalignment='center',
                                transform=ax.transAxes, fontsize=17)  # 0.0, -0.11, fontsize=12
                        # add a space to make font render correctly, convert to unicode so that there is always a string
                        box_text = unicode(get_unicode_comp(lbl2lbl[int(eval_rec.pred)], cunei_mzl_df)) + u" "
                        ax.text(0.13, -0.13, box_text, horizontalalignment='left', verticalalignment='center',
                                transform=ax.transAxes, fontproperties=font_prop, fontsize=18)  # 0.25, -0.13, fontsize=12
                    else:
                        # show det type
                        ax.text(0.0, -0.11, det_type_txt, horizontalalignment='left', verticalalignment='center',
                                transform=ax.transAxes, fontsize=17, color=box_color)
                        # show score of predicted class
                        box_text = '{:.2f}'.format(eval_rec.score)  # p(x)=
                        ax.text(0.2, -0.11, box_text, horizontalalignment='left', verticalalignment='center',
                                transform=ax.transAxes, fontsize=17)

                    if eval_rec.det_type != 2:  # not background
                        # show true class
                        ax.text(0.5, -0.11, 'GT:', horizontalalignment='left', verticalalignment='center',
                                transform=ax.transAxes, fontsize=17, color=gt_box_color)  # 0.5, -0.11, fontsize=12
                        # add a space to make font render correctly, convert to unicode so that there is always a string
                        box_text = unicode(get_unicode_comp(lbl2lbl[int(eval_rec.true)], cunei_mzl_df)) + u" "
                        ax.text(0.7, -0.13, box_text, horizontalalignment='left', verticalalignment='center',
                                transform=ax.transAxes, fontproperties=font_prop, fontsize=18)  # 0.75, -0.13, fontsize=12

        else:
            ax.axis('off')


def visualize_TP_FP(classes_to_show, list_sign_detections, dataset, col_eval_df, gt_boxes_df, customdidx2didx,
                    lbl2lbl, cunei_mzl_df, font_prop, tpfp_figure_path, context_pad=30, single_column=False,
                    show_gt_boxes=True, store_figures=True):

    # classes_to_show = [108, 96]

    # select by detection type
    select_tp = (col_eval_df.det_type == 3) & (col_eval_df.max_det)
    select_fp = ~select_tp  # & col_eval_df.pred.isin(eval_fast.list_gt_boxes_df[-1].cls.unique())
    select_loc = (col_eval_df.det_type == 0)
    select_sim = (col_eval_df.det_type == 1)
    select_bg = (col_eval_df.det_type == 2)

    # select by class
    # select_classes = col_eval_df.true.isin(classes_to_show)  # this also shows fp of other classes
    select_classes = col_eval_df.pred.isin(classes_to_show)


    # TP
    # select specific classes and only tp detections
    group_eval_df = col_eval_df[
        select_classes & select_tp].copy()  # explicit copy to add a column on slice without warning
    group_eval_df['group_idx'] = np.arange(0, len(group_eval_df))
    print("valid crops found: {}".format(len(group_eval_df)))
    # group_eval_df

    # crop sign detections
    group_crops, group_bboxes = crop_detections(group_eval_df, list_sign_detections, dataset, customdidx2didx,
                                                context_pad=context_pad, return_bboxes=True)
    len(group_crops)

    # for each sign class...
    for selected_sign_cls in classes_to_show:
        # select TP detections of class
        subgroup_eval_df = group_eval_df[group_eval_df.true == selected_sign_cls].sort_values('score', ascending=False)
        subgroup_crops = [group_crops[i] for i in subgroup_eval_df.group_idx]
        subgroup_bboxes = [group_bboxes[i] for i in subgroup_eval_df.group_idx]

        print(u'True Positives({}) for class {}[{}]'.format(len(subgroup_crops),
                                                            get_unicode_comp(lbl2lbl[selected_sign_cls], cunei_mzl_df),
                                                            selected_sign_cls))
        if len(subgroup_crops) > 0:
            sub_gt_boxes = None
            if show_gt_boxes:
                sub_gt_boxes = compute_rel_gt_boxes(gt_boxes_df, subgroup_eval_df, subgroup_bboxes, context_pad=context_pad)

            if not single_column:
                visualize_group_detections(subgroup_eval_df, subgroup_crops, lbl2lbl, cunei_mzl_df, font_prop,
                                           figs_sz=(15, 4), num_cols=6, context_pad=context_pad, max_vis=12,
                                           gt_boxes=sub_gt_boxes)  # (15, 4)
            else:
                visualize_group_detections(subgroup_eval_df, subgroup_crops, lbl2lbl, cunei_mzl_df, font_prop,
                                           figs_sz=None, num_cols=1, context_pad=context_pad, max_vis=8,
                                           gt_boxes=sub_gt_boxes)

            plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=2.0)  # prevents text from being cut  h_pad=2.0
            # how to remove border:
            # https://stackoverflow.com/questions/11837979/removing-white-space-around-a-saved-image-in-matplotlib/27227718
            plt.gca().set_axis_off()
            # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.2, wspace=0)
            # plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            if store_figures:
                plt.savefig(tpfp_figure_path.format(selected_sign_cls, "TP"), bbox_inches='tight', pad_inches=0, dpi=150)
            plt.show()

    # FP

    # select specific classes and only tp detections
    group_eval_df = col_eval_df[
        select_classes & select_fp].copy()  # explicit copy to add a column on slice without warning
    group_eval_df['group_idx'] = np.arange(0, len(group_eval_df))
    print("valid crops found: {}".format(len(group_eval_df)))
    # group_eval_df

    # crop sign detections
    group_crops, group_bboxes = crop_detections(group_eval_df, list_sign_detections, dataset, customdidx2didx,
                                                context_pad=context_pad, return_bboxes=True)
    len(group_crops)

    # for each sign class...
    for selected_sign_cls in classes_to_show:
        # select FP detections of class
        subgroup_eval_df = group_eval_df[group_eval_df.pred == selected_sign_cls].sort_values('score', ascending=False)
        subgroup_crops = [group_crops[i] for i in subgroup_eval_df.group_idx]
        subgroup_bboxes = [group_bboxes[i] for i in subgroup_eval_df.group_idx]

        print(u'False Positives({}) for class {}[{}]'.format(len(subgroup_crops),
                                                             get_unicode_comp(lbl2lbl[selected_sign_cls], cunei_mzl_df),
                                                             selected_sign_cls))

        if len(subgroup_crops) > 0:
            sub_gt_boxes = None
            if show_gt_boxes:
                sub_gt_boxes = compute_rel_gt_boxes(gt_boxes_df, subgroup_eval_df, subgroup_bboxes, context_pad=context_pad)

            if not single_column:
                visualize_group_detections(subgroup_eval_df, subgroup_crops, lbl2lbl, cunei_mzl_df, font_prop,
                                           figs_sz=(15, 4), num_cols=6, context_pad=context_pad, max_vis=12,
                                           gt_boxes=sub_gt_boxes)  # figs_sz=(15, 4)
            else:
                visualize_group_detections(subgroup_eval_df, subgroup_crops, lbl2lbl, cunei_mzl_df, font_prop,
                                           figs_sz=None, num_cols=1, context_pad=context_pad, max_vis=8,
                                           gt_boxes=sub_gt_boxes)

            plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=2.0)  # prevents text from being cut off
            # how to remove border:
            # https://stackoverflow.com/questions/11837979/removing-white-space-around-a-saved-image-in-matplotlib/27227718
            plt.gca().set_axis_off()
            # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            # plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            if store_figures:
                plt.savefig(tpfp_figure_path.format(selected_sign_cls, "FP"), bbox_inches='tight', pad_inches=0, dpi=150)
            plt.show()


