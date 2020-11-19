import pandas as pd
import numpy as np

from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt

from ast import literal_eval

import os.path

from ..alignment.LineFragment import compute_line_endpoints_by_hypo_idx
from ..detection.detection_helpers import radius_in_image
from ..detection.line_detection import line_params_from_pts, hess_normal_form_from_pts, dist_lineseg_line


class LineAnnotations(object):

    def __init__(self, collection_name, coll_scales=None, interline_dist=128/2., relative_path='../'):
        # basic paths
        self.num_classes = 2
        self.path_to_data_products = '{}data/annotations/'.format(relative_path)
        self.coll_scales = coll_scales
        self.interline_dist = interline_dist

        # load collection annotations
        self.anno_df = self.load_collection_annotations(collection_name)

        if len(self.anno_df) > 0:
            print('Load line annotations for {} dataset: {} found!'.format(collection_name,
                                                                           self.anno_df.segm_idx.nunique()))
        else:
            print('No line annotations for {} dataset'.format(collection_name))

    def load_collection_annotations(self, collection_name):
        # assemble annotation file path
        annotation_file = 'line_annotations_{}.csv'.format(collection_name)
        annotation_file_path = '{}{}'.format(self.path_to_data_products, annotation_file)

        # check if annotation file exists
        if os.path.isfile(annotation_file_path):
            # read annotation file
            anno_df = pd.read_csv(annotation_file_path, engine='python')
            # apply scale
            if self.coll_scales is not None:
                scale_vec = self.coll_scales[anno_df.segm_idx].values
                anno_df.x = (anno_df.x * scale_vec).round().astype(int)
                anno_df.y = (anno_df.y * scale_vec).round().astype(int)
            # assemble line segs
            anno_df = anno_df.groupby('segm_idx').apply(assemble_line_segments)

            ## 0) prepare meta data columns
            # add ls_x_seperate column (depends on assemble_line_segments)
            anno_df = anno_df.groupby(['segm_idx', 'line_idx']).apply(add_x_minmax)
            anno_df = anno_df.groupby('segm_idx').apply(mark_x_seperate)
            # add dist and dist_avg column
            anno_df['dist'] = anno_df.line_segs.apply(set_line_param)
            anno_df = anno_df.groupby(['segm_idx', 'line_idx']).apply(set_mean)
            ##print anno_df
            # add ls_vert_nb column (depends on assemble_line_segments)
            #anno_df = anno_df.groupby('segm_idx').apply(mark_vert_nb, self.interline_dist * 0.8)

            ## 1) group lines together
            # set inline
            #anno_df['inline'] = [np.intersect1d(*el) for el in anno_df[['ls_vert_nb', 'ls_x_separate']].values]
            #anno_df['inline'] = [np.empty(0, dtype=int)] * len(anno_df)
            anno_df['inline'] = pd.Series([np.empty(0, dtype=int)] * len(anno_df), index=anno_df.index)

            # further group line segments by order and ls_x_separate (should be respected when annotating data!)
            anno_df = anno_df.groupby('segm_idx').apply(group_ls_by_order, self.interline_dist * 5)  # * 3

            # assign actual line idx
            anno_df = anno_df.groupby('segm_idx').apply(assign_actual_line_index)

            ## 2) refine ordering
            # reset dist_avg based on gt_line_idx
            anno_df = anno_df.groupby(['segm_idx', 'gt_line_idx']).apply(set_mean)
            # assign actual line idx again
            anno_df = anno_df.groupby('segm_idx').apply(assign_actual_line_index)

            # return data frame
            return anno_df
        else:
            # return empty list (check later with len(.) to see if file exists)
            return []

    def select_df_by_segm_idx(self, segm_idx):
        assert len(self.anno_df) > 0, 'No annotations available!'
        # wrap pandas logic
        return self.anno_df[(self.anno_df.segm_idx == segm_idx)]

    def visualize_line_annotations(self, segm_idx, input_im, show_line_seg_idx=False):
        # plot line annotations
        # get segment data frame
        seg_line_df = self.select_df_by_segm_idx(segm_idx)

        # check if any anno
        if len(seg_line_df) > 0:
            # create basic plot
            fig, axes = plt.subplots(figsize=(10, 10))

            grouped = seg_line_df.groupby('line_idx')

            color = plt.cm.jet(np.linspace(0, 1, np.max(seg_line_df.line_idx) + 2))
            for i, line_rec in grouped:
                gt_line_idx = line_rec.gt_line_idx.values[0]
                line_idx = line_rec.line_idx.values[0]
                # print line_rec
                axes.plot(line_rec.x.values, line_rec.y.values, linewidth=5, color=color[gt_line_idx],)
                axes.text(line_rec.x.values[0], line_rec.y.values[0], '{}'.format(gt_line_idx),
                           bbox=dict(facecolor='blue', alpha=0.5), fontsize=8, color='white')
                if show_line_seg_idx:
                    axes.text(line_rec.x.values[1], line_rec.y.values[1], '{}'.format(line_idx),
                               bbox=dict(facecolor='red', alpha=0.5), fontsize=8, color='white')
                # axes.set_yticks([])
                # axes.set_xticks([])

            # plot last so that axis get overwritten (no need to remove ticks :)
            axes.imshow(input_im, cmap='gray')
            plt.show()

    def get_hypo_line_labeling_for_segm(self, segm_idx, line_hypos_agg, verbose=False):

        # select line segment ground truth
        seg_ls_df = self.select_df_by_segm_idx(segm_idx).copy()
        # from n points only n-1 segments -> remove empty ones
        seg_ls_df = seg_ls_df[seg_ls_df.line_segs.apply(len) > 0]

        # check if any annotations found
        if len(seg_ls_df) > 0:
            # assign hypo lines to gt line segments
            gt_line_segs = seg_ls_df.line_segs.values.tolist()
            gt_ls_lbl, gt_ls_dist = assign_lines_to_gt_line_segments(gt_line_segs, line_hypos_agg)
            # update dataframe
            seg_ls_df['hypo_line_lbl'] = gt_ls_lbl
            seg_ls_df['hypo_line_dist'] = np.sqrt(gt_ls_dist)
            # decide hypo line labels
            seg_ls_df = seg_ls_df.groupby(['gt_line_idx']).apply(decide_hypo_line_lbl)
        else:
            if verbose:
                print('No line ground truth available for segment idx [{}]!'.format(segm_idx))

        return seg_ls_df

    def get_assignment_for_line_hypos(self, segm_idx, line_hypos_agg):
        # create empty dummy for cases where no annotations available
        gt_line_assignment = pd.DataFrame()

        if len(self.anno_df) > 0:
            # get labelling
            seg_ls_df = self.get_hypo_line_labeling_for_segm(segm_idx, line_hypos_agg)

            if len(seg_ls_df) > 0:
                # in case of multiple annotations per hypo line, pick the one with smallest distance
                gt_line_assignment = seg_ls_df.sort_values('hypo_line_dist').groupby('hypo_line_lbl').head(1)[
                    ['gt_line_idx', 'hypo_line_lbl']]
                gt_line_assignment = gt_line_assignment.sort_values('gt_line_idx')

        return gt_line_assignment

    def visualize_hypo_line_assignments(self, segm_idx, line_hypos_agg, input_im):
        # get labelling
        seg_ls_df = self.get_hypo_line_labeling_for_segm(segm_idx, line_hypos_agg)
        gt_ls_lbl = seg_ls_df.hypo_line_lbl.values
        gt_line_segs = seg_ls_df.line_segs.values

        # visualize
        visualize_line_segments_with_labels(gt_line_segs, gt_ls_lbl, input_im)

    def visualize_gt_lines_with_assignments(self, segm_idx, line_hypos_agg, center_im):

        # gt assignment
        gt_line_assignment = self.get_assignment_for_line_hypos(segm_idx, line_hypos_agg)

        # get labelling
        seg_ls_df = self.get_hypo_line_labeling_for_segm(segm_idx, line_hypos_agg)
        gt_ls_lbl = seg_ls_df.gt_line_idx.values
        gt_line_segs = seg_ls_df.line_segs.values

        # get line hypo endpoints
        list_hypo_endpts = [np.fliplr(np.array(compute_line_endpoints_by_hypo_idx(hidx, line_hypos_agg)).
                                      reshape(2, 2)).ravel() for hidx in gt_line_assignment.hypo_line_lbl.values]

        # get color map
        color = plt.cm.spectral(np.linspace(0, 1, np.max(gt_ls_lbl) + 1))  # len(np.unique(gt_ls_lbl))

        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        ax = axes.ravel()

        ax[0].imshow(center_im, cmap='gray')
        ax[0].set_title('Input image')

        ax[1].imshow(center_im * 0)
        for line, li in zip(gt_line_segs, gt_ls_lbl):
            p0, p1 = line
            ax[1].plot((p0[0], p1[0]), (p0[1], p1[1]), color=color[li], linewidth=2)

        ax[1].set_xlim((0, center_im.shape[1]))
        ax[1].set_ylim((center_im.shape[0], 0))
        ax[1].set_title('gt line segments and assigned line hypos')

        for idx, line_pts in enumerate(list_hypo_endpts):
            ax[1].plot(line_pts[::2], line_pts[1::2], '-', color=color[int(idx)], linewidth=2)
            ax[1].text(line_pts[0], line_pts[1], '{}'.format(idx),
                       bbox=dict(facecolor='blue', alpha=0.5), fontsize=8, color='white')



#### HELPERS

# create line segment column

def assemble_line_segments(group):
    # assemble line segments
    line_grouped = group.groupby('line_idx')
    line_segs = []
    # iterate over lines
    for lidx, lgroup in line_grouped:
        num_pts = len(lgroup)
        # iterate over segments
        for sidx in range(num_pts):
            # assemble segments
            if sidx == num_pts - 1:
                line_segs.append(())
            else:
                line_segs.append(((lgroup.iloc[sidx].x, lgroup.iloc[sidx].y),
                                  (lgroup.iloc[sidx + 1].x, lgroup.iloc[sidx + 1].y)
                                  ))
    # assign to group
    group['line_segs'] = line_segs

    return group


# group line segments to line

def add_x_minmax(group):
    group['xmin'] = group.x.min()
    group['xmax'] = group.x.max()
    return group


def mark_x_seperate(group):
    # iterate line segments
    list_left_or_right = []
    for i, (ls_idx, line_seg) in enumerate(group.iterrows()):
        # create list of segments to the left
        index_left = group.line_idx[group.xmax < line_seg.xmin].unique()
        # create list of segments to the left
        index_right = group.line_idx[group.xmin > line_seg.xmax].unique()
        # concat and append to list
        list_left_or_right.append(np.concatenate([np.array(index_left), np.array(index_right)]))
    group['ls_x_separate'] = list_left_or_right
    return group


def set_line_param(line_seg):

    if len(line_seg) > 0:
        # use basic line equation
        #line_params = line_params_from_pts(line_seg[0], line_seg[1])

        # use hess normal form (in corporates angle)
        line_params = hess_normal_form_from_pts(line_seg[0], line_seg[1])
        return line_params[1]  # only interest in height
    else:
        return np.NaN


def set_mean(group):
    group['dist_avg'] = group.dist.mean()
    return group


def mark_vert_nb(group, interline_thresh):
    # iterate line segments
    list_vert_nb = []
    for i, (ls_idx, line_seg) in enumerate(group.iterrows()):
        # create list of segments to the left
        index_vert_near = group.line_idx[(group.dist >= 0) &
                                         (np.abs(group.dist_avg - line_seg.dist_avg) < interline_thresh)].unique()
        list_vert_nb.append(np.array(index_vert_near))

    group['ls_vert_nb'] = list_vert_nb
    return group


# def make_inline_symmetric(group):
#     # iterate over line segments, and make symmetric reference of inline
#     for i, (sidx, line_seg) in enumerate(group.iterrows()):
#         if len(line_seg.inline) > 0:
#             select_inline = group.line_idx.isin(line_seg.inline)
#             group.loc[select_inline, 'inline'] = select_inline.sum() * [line_seg.inline]
#     # deal with type mismatch in column (did find no better way :/)
#     inline_list = []
#     for el in group.inline.astype(list).values:
#         if isinstance(el, np.ndarray):
#             inline_list.append(el)
#         else:
#             inline_list.append(np.array([el]))
#     group['inline'] = inline_list
#     # return
#     return group


def group_ls_by_order(group, interline_thresh):
    last_lidx = -1
    last_xseparate = []
    # QUICK FIX: use this to deal with loc and list inserts (loc[idx] works rather than loc[idx, col]!!)
    group_inline = group.inline
    # iter line_idx aggregate
    # https://stackoverflow.com/questions/20067636/pandas-dataframe-get-first-row-of-each-group/49148885#49148885
    ls_agg = group.sort_values('line_idx').groupby('line_idx').nth(0)  #.first() is dangerous

    for curr_lidx, ls_agg_rec in ls_agg.iterrows():
        if last_lidx != -1:
            # check if last line segment is x separate
            if np.any(np.isin(ls_agg_rec.ls_x_separate, last_lidx)):
                last_rec = ls_agg.loc[last_lidx]
                # check if last line segment on the left
                ls_left = (last_rec.xmax < ls_agg_rec.xmin)
                if ls_left:
                    # check if vertical distance is small
                    vert_dist_is_small = np.abs(last_rec.dist_avg - ls_agg_rec.dist_avg) < interline_thresh
                    if vert_dist_is_small:
                        # check if already inline
                        if last_lidx not in ls_agg_rec.inline:
                            # print('merge line segments {} with {}'.format(curr_lidx, last_lidx))
                            # create new inlines
                            # do not use ls_agg_rec.inline, since it does not get updated during loop
                            #new_inline = np.concatenate([ls_agg_rec.inline, np.array([last_lidx])])
                            #new_last_inline = np.concatenate([ls_agg_rec.inline, np.array([curr_lidx])])
                            new_inline = np.concatenate([group_inline.loc[group.line_idx == curr_lidx].values[0], np.array([last_lidx])])
                            new_last_inline = np.concatenate([group_inline.loc[group.line_idx == last_lidx].values[0], np.array([curr_lidx])])
                            # add to data frame (loc[idx] works rather than loc[idx, col]!!)
                            select_line_idx = (group.line_idx == curr_lidx)
                            group_inline.loc[select_line_idx] = [new_inline] * select_line_idx.sum()
                            select_line_idx = (group.line_idx == last_lidx)
                            group_inline.loc[select_line_idx] = [new_last_inline] * select_line_idx.sum()

        # set last values
        last_lidx = curr_lidx
        last_xseparate = ls_agg_rec.ls_x_separate
    return group


# finalize assignment

def assign_actual_line_index(group):
    # create new column
    group['gt_line_idx'] = np.ones(len(group), dtype=int) * -1
    # iterate over line segments and assign acutal_line_idx (segs sorted by 1) y position 2) x position)
    new_idx = 0
    for sidx, line_seg in group.sort_values(['dist_avg', 'x']).iterrows():
        # check if index is already set
        if group.loc[sidx, 'gt_line_idx'] == -1:
            # assign index to line segment
            group.loc[group.line_idx == line_seg.line_idx, 'gt_line_idx'] = new_idx
            # assign same index to inline segments
            for lidx in line_seg.inline:
                group.loc[group.line_idx == lidx, 'gt_line_idx'] = new_idx
            # finally increment index
            new_idx += 1
        # if index is already set, extend it to all inline members
        else:
            curr_idx = group.loc[sidx, 'gt_line_idx']
            # assign same index to inline segments
            for lidx in line_seg.inline:
                group.loc[group.line_idx == lidx, 'gt_line_idx'] = curr_idx

    return group


# for eval need to assign detection lines to ground truth lines

def assign_lines_to_gt_line_segments(gt_line_segs, line_hypos_agg):
    # get line pts from polar lines
    line_pts = []
    for idx in range(len(line_hypos_agg)):
        # compute line endpoints
        line_pts.append(compute_line_endpoints_by_hypo_idx(idx, line_hypos_agg))
        #line_pts.append(line_frag.compute_line_endpoints(-1, hypo_idx=i))

    line_pts = np.vstack(line_pts)
    line_pts = np.flip(line_pts.reshape((-1, 2, 2)), axis=2).reshape(-1, 4)

    # get line segments
    line_seg_pts = np.stack(gt_line_segs).reshape(len(gt_line_segs), -1)

    # compute distance between line segments and lines
    X2_dist = cdist(line_pts, line_seg_pts,
                    lambda lpts, spts: dist_lineseg_line(spts[:2], spts[2:], lpts[:2], lpts[2:]))
    # assign line segments to nearest line
    ls_labels = np.argmin(X2_dist, axis=0)
    ls_dist = np.min(X2_dist, axis=0)

    return ls_labels, ls_dist


def decide_hypo_line_lbl(group):
    # count hypo line labels
    uv, counts = np.unique(group.hypo_line_lbl, return_counts=True)
    # get idx to all largest
    largest_select = (np.max(counts) == counts)
    # check if tiebreak is required
    if largest_select.sum() > 1:
        # for each similar large group compute mean hypo_line_dist and pick largest
        tiebreak_df = group.groupby('hypo_line_lbl').hypo_line_dist.mean()
        most_freq_hypo_lbl = tiebreak_df[uv[largest_select]].idxmax()
    else:
        most_freq_hypo_lbl = uv[np.argmax(counts)]

    # assign most frequent label
    group['hypo_line_lbl'] = most_freq_hypo_lbl

    return group


# visualize

def visualize_line_segments_with_labels(gt_line_segs, gt_ls_lbl, center_im, line_hypo_endpts=None):

    color = plt.cm.spectral(np.linspace(0, 1, np.max(gt_ls_lbl) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    ax = axes.ravel()

    ax[0].imshow(center_im, cmap='gray')
    ax[0].set_title('Input image')

    # ax[1].imshow(lbl_ind_x, cmap='gray')
    # ax[1].set_title('line det')

    ax[1].imshow(center_im * 0)
    for line, li in zip(gt_line_segs, gt_ls_lbl):
        p0, p1 = line
        ax[1].plot((p0[0], p1[0]), (p0[1], p1[1]), color=color[li], linewidth=2)
        ax[1].text(p0[0], p0[1], '{}'.format(li),
                   bbox=dict(facecolor='blue', alpha=0.5), fontsize=8, color='white')
    ax[1].set_xlim((0, center_im.shape[1]))
    ax[1].set_ylim((center_im.shape[0], 0))
    ax[1].set_title('gt line segments and assigned line hypos')

    if line_hypo_endpts is not None:
        for idx, line_pts in enumerate(line_hypo_endpts):
            ax[1].plot(line_pts[::2], line_pts[1::2], '-', color=color[int(idx)], linewidth=2)


