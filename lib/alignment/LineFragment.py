import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
from matplotlib.collections import PatchCollection

from scipy.spatial.distance import cdist
from scipy.ndimage import distance_transform_edt
from sklearn.linear_model import LinearRegression

from skimage.draw import line, line_aa, polygon, polygon_perimeter
from skimage.measure import grid_points_in_poly, points_in_poly
from skimage.color import label2rgb
from skimage.morphology import skeletonize

from .LineMatching1D import LineMatching1D
from ..detection.line_detection import compute_image_label_map, line_params_from_pts, line_pts_from_polar_line, \
    clip_bbox_using_line, clip_bbox_using_line_segmentation
from ..detection.detection_helpers import coord_in_image, radius_in_image, nms
from ..evaluations.sign_tl_evaluation import compute_accuracy, convert_alignments_for_eval, compute_levenshtein
from ..evaluations.sign_tl_evaluation import compute_bleu_score as compute_bleu  # due to name conflict
from ..utils.bbox_utils import convert_bbox_local2global


# line fragment helpers

def compute_line_points(angle, dist, lbl_map_shape=[100, 100]):
    # model is in lbl_map_shape and will be up-scaled according to net arch
    # compute start and end point of line segment
    pt1, pt2 = line_pts_from_polar_line(angle, dist, x1=lbl_map_shape[1])
    ipt1 = np.rint(coord_in_image(np.concatenate(pt1))).astype(int)
    ipt2 = np.rint(coord_in_image(np.concatenate(pt2))).astype(int)

    return [ipt1[1], ipt1[0], ipt2[1], ipt2[0]]


def compute_line_endpoints_by_hypo_idx(hypo_idx, line_hypos_agg):
    # hypo_idx: use line_hypos_agg idx to select line_hypo
    line_rec = line_hypos_agg.iloc[[hypo_idx]]
    # compute line endpoints
    line_pts = compute_line_points(line_rec.angle.values, line_rec.dist.values)
    return line_pts


def compute_line_polygon(angle, dist, lbl_map_shape, ortho_pad=1.0):
    # compute line parallelogram with thickness 2 * ortho_pad
    # return: 4 x 2 vertices in counter-clockwise order

    # get upper bound
    pt1, pt2 = line_pts_from_polar_line(angle, dist - ortho_pad, x1=lbl_map_shape[1])
    ipt1u = np.rint(coord_in_image(np.concatenate(pt1))).astype(int)
    ipt2u = np.rint(coord_in_image(np.concatenate(pt2))).astype(int)
    # get lower bound
    pt1, pt2 = line_pts_from_polar_line(angle, dist + ortho_pad, x1=lbl_map_shape[1])
    ipt1l = np.rint(coord_in_image(np.concatenate(pt1))).astype(int)
    ipt2l = np.rint(coord_in_image(np.concatenate(pt2))).astype(int)
    # collect points
    r = np.stack([ipt1l[0], ipt2l[0], ipt2u[0], ipt1u[0]])
    c = np.stack([ipt1l[1], ipt2l[1], ipt2u[1], ipt1u[1]])

    return np.stack([c, r], axis=1)


def compute_bbox_ctr(bboxes):
    aligned_ctrs = np.zeros((bboxes.shape[0], 2))
    aligned_ctrs[:, 0] = (bboxes[:, 2] + bboxes[:, 0]) / 2
    aligned_ctrs[:, 1] = (bboxes[:, 3] + bboxes[:, 1]) / 2
    return aligned_ctrs


def update_detections_array(dets_arr, det_boxes):
    # compute new ctr
    ctr_col = compute_bbox_ctr(det_boxes)
    # get ID column
    id_col = dets_arr[:, 0].reshape(-1, 1)
    # score column
    score_col = dets_arr[:, 3].reshape(-1, 1)
    # global idx column
    gid_col = dets_arr[:, -1].reshape(-1, 1)
    # stack together
    # [ID, cx, cy, score, x1, y1, x2, y2, idx]
    new_dets_arr = np.hstack([id_col, ctr_col, score_col, det_boxes, gid_col])

    return new_dets_arr


def compute_section_hypo_vec(tl_line_rec, list_sidx, section_xmin, section_xmax, a, b):
    section_rec = tl_line_rec.loc[list_sidx]
    # compute lengths
    section_len = section_rec.prior_sign_width.sum()
    det_section_len = section_xmax - section_xmin
    # compute offset
    sign_xpos = section_rec.prior_sign_width.cumsum() - section_rec.prior_sign_width / 2.
    # compute offsets
    vec_x = section_xmin + (sign_xpos * det_section_len / section_len)
    vec_y = (lambda x: a * x + b)(vec_x)
    hypo_ctrs = np.stack([vec_x, vec_y], axis=1)
    if True:  # enable filtering of too small or big gaps
        ## print("det_len:", det_section_len, 'sec len:', section_len)
        # mark if too small
        if det_section_len < section_len * 0.2:  # 0.25
            hypo_ctrs = np.ones_like(hypo_ctrs) * -1
        # mark if too big
        if det_section_len > section_len * 5.0:  # 4.0
           hypo_ctrs = np.ones_like(hypo_ctrs) * -1
        # mark damaged signs
        #hypo_ctrs[section_rec.status != 1] = -1
    return hypo_ctrs


def plot_boxes(boxes, confidence=None, ax=None, color=0.8):
    # handle input
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 12))
    if confidence is None:
        confidence = np.ones(boxes.shape[0]) * color  # 0.2
    # plot
    for ii, bbox in enumerate(boxes):
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor=plt.cm.plasma(confidence[ii]),
                          alpha=0.8,
                          # alpha=cnn_confidence[ii],
                          linewidth=2.0))  # edgecolor='red'


# CLASS


class LineFragment(object):

    def __init__(self, line_hypos, line_lbl_segments, tl_df, stats, input_im, detections=None):
        self.line_hypos = line_hypos
        self.line_hypos_agg = line_hypos.groupby('label').mean()

        # group lines and remember index (still needed?)
        self.group2line = line_hypos.groupby('label').mean().index.values
        # compute interline distance median
        self.dist_interline_median = line_hypos['group_diff'].median()

        self.line_lbl_segments = line_lbl_segments

        self.box_cmap = plt.get_cmap('nipy_spectral')  # nipy_spectral, Spectral

        self.tl_df = tl_df
        self.stats = stats

        self.detections = detections
        self.line_det_shape = line_lbl_segments.shape  # lbl_ind.shape
        self.input_im = input_im
        self.input_shape = input_im.shape

    def check_line_idx_in_bounds(self, tl_line_idx, tl_line_idx_cmp):
        if not np.any(self.tl_df.line_idx == tl_line_idx_cmp):
            raise KeyError('No tl available at idx {}!'.format(tl_line_idx_cmp))
        if not np.any(self.line_hypos_agg.tl_line == tl_line_idx):
            raise KeyError('No line available at idx {}!'.format(tl_line_idx))

    # access line_hypos and transliteration df

    def get_tl_rec(self, tl_line_idx):
        return self.tl_df[self.tl_df.line_idx == tl_line_idx]

    def get_line_rec(self, tl_line_idx):
        # get line hypo record from line_hypos_agg
        line_rec = self.line_hypos_agg[self.line_hypos_agg.tl_line == tl_line_idx]
        # deal with the case multiple matching lines
        assert len(line_rec) <= 1, 'Error in tl_line assignment: multiple lines assigned!'
        return line_rec

    # get assigned lines
    def get_assigned_lines_idx(self):
        # return tl_line indices which have been matched with line_hypos
        assigned_tl_indices = self.line_hypos_agg.sort_values('tl_line').tl_line.unique()
        assigned_tl_indices = assigned_tl_indices[assigned_tl_indices >= 0]
        # make sure that assigned_tl_indices are not longer than tl_df
        if len(assigned_tl_indices) > len(self.tl_df.line_idx.unique()):
            assigned_tl_indices = assigned_tl_indices[:len(self.tl_df.line_idx.unique())]
        return assigned_tl_indices

    # get assignment space (cartesian product of tl_line_indices and hypo_line_indices)
    def get_alignment_space(self):
        tl_line_indices = self.tl_df.line_idx.unique()
        # hypo_line_indices = self.line_hypos_agg.index.sort_values().values
        # why? should just use index of line_hypos_agg
        hypo_line_tmp = self.line_hypos_agg.tl_line
        hypo_line_indices = hypo_line_tmp[hypo_line_tmp >= 0].sort_values().unique()
        return hypo_line_indices, tl_line_indices

    # fast and compact representations of line models

    def compute_line_endpoints(self, tl_line_idx, hypo_idx=None):
        # hypo_idx: use line_hypos_agg idx to select line_hypo

        # get line record
        if hypo_idx is None:
            line_rec = self.get_line_rec(tl_line_idx)
        else:
            line_rec = self.line_hypos_agg.iloc[[hypo_idx]]

        if len(line_rec) > 0:
            # compute line endpoints
            line_pts = compute_line_points(line_rec.angle.values, line_rec.dist.values, self.line_det_shape)
            return line_pts
        else:
            return []

    def compute_line_polygon(self, tl_line_idx, ortho_pad=None):
        if ortho_pad is None:
            ortho_pad = self.dist_interline_median / 2.
        # get line record
        line_rec = self.get_line_rec(tl_line_idx)
        if len(line_rec) > 0:
            # compute line polygon
            line_poly = compute_line_polygon(line_rec.angle.values, line_rec.dist.values,
                                             self.line_det_shape, ortho_pad=ortho_pad)
            return line_poly
        else:
            return []

    # useful, but "heavy" representations (pre-compute)

    def compute_line_region_mask(self, tl_line_idx, ortho_pad=None):
        if ortho_pad is None:
            ortho_pad = self.dist_interline_median / 2.
        # get line record
        line_rec = self.get_line_rec(tl_line_idx)
        if len(line_rec) > 0:
            # compute line polygon
            line_poly = compute_line_polygon(line_rec.angle.values, line_rec.dist.values,
                                             self.line_det_shape, ortho_pad=ortho_pad)
            # compute line mask
            line_mask = grid_points_in_poly(self.input_shape, line_poly)
            # compute line mask (a bit slower)
            # rr, cc = polygon(line_poly[:,0], line_poly[:,1], shape=self.input_shape)
            # line_mask = np.zeros(self.input_shape).astype(bool)
            # line_mask[rr, cc] = True
            return line_mask
        else:
            return []

    def compute_line_region_border(self, tl_line_idx, ortho_pad=None):
        if ortho_pad is None:
            ortho_pad = self.dist_interline_median / 2.
        # get line record
        line_rec = self.get_line_rec(tl_line_idx)
        if len(line_rec) > 0:
            # compute line polygon
            line_poly = compute_line_polygon(line_rec.angle.values, line_rec.dist.values,
                                             self.line_det_shape, ortho_pad=ortho_pad)
            # compute line mask
            rr, cc = polygon_perimeter(line_poly[:, 0], line_poly[:, 1], shape=self.input_shape)
            line_border = np.zeros(self.input_shape, dtype=bool)
            line_border[rr, cc] = True
            return line_border
        else:
            return []

    # fast detection filter (depends on number of detections (e.g. strength of NMS)

    def get_region_detections(self, tl_line_idx, ortho_pad=None):
        # selects detections from box around line with dist_interline_median height
        line_poly = self.compute_line_polygon(tl_line_idx, ortho_pad=ortho_pad)
        if len(line_poly) > 0:
            # check if points inside polygon
            in_range_boxes = points_in_poly(np.fliplr(self.detections[:, 1:3]), line_poly)
            # [ID, cx, cy, score, x1, y1, x2, y2, idx]
            return self.detections[in_range_boxes, :]
        else:
            return []

    # access line segmentation mask

    def get_line_segmentation(self, tl_line_idx, skeleton=False):
        # check if there is a line model
        line_rec = self.get_line_rec(tl_line_idx)
        if len(line_rec) > 0:
            line_segment = self.line_lbl_segments == line_rec.index.values + 1  # segment labels are +1
            # check if there is any segmented pixel
            if np.any(line_segment):
                # skeletonize lbl map if required
                if skeleton:
                    line_segment = skeletonize(line_segment)

                # map from lblmap shape to image resolution
                return compute_image_label_map(line_segment, self.input_shape)
            else:
                return []
        else:
            return []

    def get_line_segmentation_bbox(self, tl_line_idx):
        line_rec = self.get_line_rec(tl_line_idx)
        if len(line_rec) > 0:
            line_segment = self.line_lbl_segments == line_rec.index.values + 1  # segment labels are +1
            seg_xs, seg_ys = np.nonzero(line_segment)
            # check if segment available
            if len(seg_xs) > 0:
                # assemble segment bbox
                bbox = [np.min(seg_ys), np.min(seg_xs),  np.max(seg_ys), np.max(seg_xs)]
                # map to image resolution
                return map(coord_in_image, bbox)
            else:
                # fall back to model bounding box
                line_pts = self.compute_line_endpoints(tl_line_idx)
                # assemble segment bbox
                bbox = [np.min(line_pts[1::2]), np.min(line_pts[::2]), np.max(line_pts[1::2]), np.max(line_pts[::2])]
                return bbox
        else:
            return []

    # detection refinement using line model

    def refine_detections_using_line_hypos(self, tl_line_idx, region_detections, min_dist=128 / 1.6,
                                           use_segmentation=False):
        # iterate alignments and refine detection bbox
        # * use inter_line_dist to limit height of bbox
        # Attention: Will produce "line" if applied to bbox that is completely out of line scope
        # however, since region detections are used that should not be a problem

        # get line endpoints using tl_line_idx
        line_pts = self.compute_line_endpoints(tl_line_idx)
        line_pts_arr = np.fliplr(np.array(line_pts).reshape((2, 2)))  # pretty format
        # line_rec = self.get_line_rec(tl_line_idx)

        # select skeletonized line segmentation
        line_skeleton = []
        if use_segmentation:
            line_skeleton = self.get_line_segmentation(tl_line_idx, skeleton=True)

        bboxes = region_detections[:, 4:8]
        list_bbox = []
        for bbox in bboxes:
            if use_segmentation and len(line_skeleton) > 0:
                new_bbox = clip_bbox_using_line_segmentation(bbox, line_pts_arr, line_skeleton, min_dist=min_dist)
            else:
                new_bbox = clip_bbox_using_line(bbox, line_pts_arr, min_dist=min_dist)
            list_bbox.append(new_bbox)

        return np.stack(list_bbox)

    # visualize line in different ways

    def visualize_line(self, tl_line_idx):
        line_pts = self.compute_line_endpoints(tl_line_idx)
        if len(line_pts) > 0:
            plt.plot(line_pts[1::2], line_pts[::2], linewidth=4)
            plt.imshow(self.input_im, cmap='gray')

    def visualize_region_mask(self, tl_line_idx, ortho_pad=None):
        line_mask = self.compute_line_region_mask(tl_line_idx, ortho_pad=ortho_pad)
        if len(line_mask) > 0:
            plt.imshow(line_mask, cmap='gray')

    def visualize_region(self, tl_line_idx, ortho_pad=None):
        line_mask = self.compute_line_region_mask(tl_line_idx, ortho_pad=ortho_pad)
        if len(line_mask) > 0:
            image_label_overlay = label2rgb(line_mask, image=self.input_im)
            plt.imshow(image_label_overlay, cmap='gray')

    def visualize_region_detections(self, tl_line_idx, min_conf=0, ortho_pad=None, show_boxes=False, refined=False):
        # select line region detections
        region_detections = self.get_region_detections(tl_line_idx, ortho_pad=ortho_pad)
        if len(region_detections) > 0:
            # select min confidence detections
            to_show = region_detections[:, 3] >= min_conf
            signs = np.vstack((region_detections[to_show, 1:3], [0, 0]))
            # plot points
            plt.plot(signs[:, 0], signs[:, 1], 'bo', markersize=6, color='green')
            plt.imshow(self.input_im, cmap='gray')
            if show_boxes:
                if refined:
                    det_boxes = self.refine_detections_using_line_hypos(tl_line_idx, region_detections[to_show, :],
                                                                        use_segmentation=True,
                                                                        min_dist=128 / 2.)
                else:
                    det_boxes = region_detections[to_show, 4:8]
                plot_boxes(det_boxes, confidence=None, ax=plt.gca())

    def tab_visualize_detections(self, region_detections=[], min_conf=0, show_boxes=False, color=0.9):
        # select line region detections
        if len(region_detections) == 0:
            region_detections = self.detections
        # select min confidence detections
        to_show = region_detections[:, 3] >= min_conf
        signs = np.vstack((region_detections[to_show, 1:3], [0, 0]))
        # plot points
        plt.plot(signs[:, 0], signs[:, 1], 'bs', markersize=6, color=self.box_cmap(0.7))  # color='green'
        plt.imshow(self.input_im, cmap='gray')
        if show_boxes:
            det_boxes = region_detections[to_show, 4:8]
            plot_boxes(det_boxes, confidence=None, ax=plt.gca(), color=color)

    def visualize_segmentation(self, tl_line_idx):
        line_segment = self.get_line_segmentation(tl_line_idx)
        if len(line_segment) > 0:
            image_label_overlay = label2rgb(line_segment, image=self.input_im)
            plt.imshow(image_label_overlay, cmap='gray')

    def visualize_sign_hypos(self, sign_hypos):
        # universal visualize list of Nx2 coordinates
        #plt.plot(sign_hypos[:, 0], sign_hypos[:, 1], 'wo', markersize=6)
        plt.plot(sign_hypos[:, 0], sign_hypos[:, 1], 'd', color=self.box_cmap(0.25), markersize=9, label='null hypo')
        plt.plot(sign_hypos[:, 0], sign_hypos[:, 1], color=self.box_cmap(0.25), linewidth=1.5)

        plt.imshow(self.input_im, cmap='gray')

    def visualize_null_hypo_alignments(self, tl_line_idx, tl_line_idx_cmp=None, max_dist_thresh=1,
                                       show_search_field=True, show_alignments=True):
        # visualize alignments achieve using null hypo segmentation
        # TODO: handle mutliple null hypo versions like basic
        # allow (tl_line_idx != tl_line_idx_cmp)
        # tl_line_idx --> choice of line detection (line model, segmentation, detections)
        # tl_line_idx_cmp --> choice of transliteration line (tl_line_rec)
        if tl_line_idx_cmp is None:
            tl_line_idx_cmp = tl_line_idx
        self.check_line_idx_in_bounds(tl_line_idx, tl_line_idx_cmp)

        if show_alignments:
            # get alignment index and detections
            (score, alignments, region_det, sign_hypos, tl_line_rec) \
                = self.compute_ransac_score(tl_line_idx, tl_line_idx_cmp, max_dist_thresh=max_dist_thresh,
                                            return_alignments=True)
            # select detections using alignment index
            aligned = region_det[alignments[alignments >= 0], 1:3]
            # plot
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.plot(aligned[:, 0], aligned[:, 1], 's', color=self.box_cmap(0.6), markersize=8)
        else:
            # get hypothesis using tl_line_idx and tl_line_idx_cmp
            sign_hypos = self.create_null_hypo_segmentation(tl_line_idx, tl_line_idx_cmp)
            # get unique labels from transliteration using tl_line_idx_cmp
            tl_line_rec = self.get_tl_rec(tl_line_idx_cmp)
            # plot
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        ax.plot(sign_hypos[:, 0], sign_hypos[:, 1], 'd', color=self.box_cmap(0.25), markersize=9, label='null hypo')
        ax.imshow(self.input_im, cmap='gray')

        if show_search_field:
            patches = []
            for x, lbl in zip(sign_hypos, tl_line_rec.lbl.values):
                # circle = ptc.Circle((x[0], x[1]), radius=self.parent.tl.tblSignHeight/2, fc='g')
                # patches.append(circle)
                if False:
                    if lbl in self.stats.signs.keys():
                        sign_width = self.stats.signs[lbl]["median"]["width"]
                    else:
                        sign_width = 1
                else:
                    sign_width = self.stats.get_sign_width(lbl, sign_width=1)
                # if only theta is used this is enough
                eig_vals = np.array([sign_width, 1]) * max_dist_thresh
                # create ellipse
                scaled_ev = eig_vals * self.stats.tblSignHeight
                ellipse = ptc.Ellipse((x[0], x[1]), scaled_ev[0], scaled_ev[1], angle=0, fc='g')
                patches.append(ellipse)

            p = PatchCollection(patches, alpha=0.25)
            ax.add_collection(p)

    def visualize_imputed_signs(self, tl_line_idx, tl_line_idx_cmp=None, anno_df=None, min_dets_inline=1, show_null_hypo=True):
        # visualize null or cond hypos
        # allow (tl_line_idx != tl_line_idx_cmp)
        # tl_line_idx --> choice of line detection (line model, segmentation, detections)
        # tl_line_idx_cmp --> choice of transliteration line (tl_line_rec)
        if tl_line_idx_cmp is None:
            tl_line_idx_cmp = tl_line_idx
        self.check_line_idx_in_bounds(tl_line_idx, tl_line_idx_cmp)

        # plot region detections (these should be refined/aligned detections, if used for sign imputation)
        # region detections
        region_det = self.get_region_detections(tl_line_idx)
        # select detections using alignment index
        aligned = region_det[:, 1:3]
        # plot ctr points
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.plot(aligned[:, 0], aligned[:, 1], 's', color=self.box_cmap(0.6), markersize=8)

        # plot inputed signs
        if show_null_hypo:
            # get hypothesis using tl_line_idx and tl_line_idx_cmp
            l_lbls, l_boxctrs, l_boxes = self.create_null_hypo_alignments(tl_line_idx, tl_line_idx_cmp)
        else:
            (l_hypos, l_boxes, l_anno_idx,
             l_meta) = self.create_conditional_hypo_alignments(tl_line_idx, tl_line_idx_cmp, anno_df,
                                                               min_dets_inline, ncompl_thresh=-1)
        plot_boxes(l_boxes, confidence=None, ax=plt.gca())

        # finish plot
        if not show_null_hypo:
            # plot detections again in different color
            plot_boxes(region_det[:, 4:8], confidence=None, ax=plt.gca(), color=0.1)
        ax.set_title("# ref detections: {} | # train detections: {}".format(len(aligned), len(l_boxes)))
        ax.imshow(self.input_im, cmap='gray')

    def visualize_gm_alignments(self, tl_line_idx, tl_line_idx_cmp=None):
        # allow (tl_line_idx != tl_line_idx_cmp)
        # tl_line_idx --> choice of line detection (line model, segmentation, detections)
        # tl_line_idx_cmp --> choice of transliteration line (tl_line_rec)
        if tl_line_idx_cmp is None:
            tl_line_idx_cmp = tl_line_idx
        self.check_line_idx_in_bounds(tl_line_idx, tl_line_idx_cmp)

        # get alignment index and detections
        (score, region_det, sign_hypos, gm_tl_line_rec) \
            = self.compute_line_matching_score(tl_line_idx, tl_line_idx_cmp, return_alignments=True)
        # select detections using alignment index
        alignments = gm_tl_line_rec.region_det_idx.values
        aligned = region_det[alignments[alignments >= 0], 1:3]

        self.gm_tl_line_rec = gm_tl_line_rec

        # plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        # plot null hypo coordinates and line
        ax.plot(sign_hypos[:, 0], sign_hypos[:, 1], 'd', color=self.box_cmap(0.25), markersize=9, label='null hypo')
        ax.plot(sign_hypos[:, 0], sign_hypos[:, 1], color=self.box_cmap(0.25), linewidth=1.5)
        # plot aligned detections and line
        ax.plot(aligned[:, 0], aligned[:, 1], 's', color=self.box_cmap(0.6), markersize=10, label='gm aligned detections')
        ax.plot(aligned[:, 0], aligned[:, 1], color=self.box_cmap(0.6), linewidth=2.0)

        # annotate
        for i, pos_idx in enumerate(gm_tl_line_rec.iloc[alignments >= 0].pos_idx.values):
            ax.annotate(pos_idx, (aligned[i, 0], aligned[i, 1]), fontsize=10, weight='bold',
                        color=plt.cm.gray(0.05), va='center', ha='center')

        # plot tablet
        ax.imshow(self.input_im, cmap='gray')
        #ax.legend(fancybox=True, loc='best')
        ax.axis('off')

    # create sign hypothesis based on line model and its assigned transliteration

    def create_null_hypo_basic(self, tl_line_idx, tl_line_idx_cmp=None):
        # allow (tl_line_idx != tl_line_idx_cmp)
        # tl_line_idx --> choice of line detection (line model, segmentation, detections)
        # tl_line_idx_cmp --> choice of transliteration line (tl_line_rec)
        if tl_line_idx_cmp is None:
            tl_line_idx_cmp = tl_line_idx
        self.check_line_idx_in_bounds(tl_line_idx, tl_line_idx_cmp)

        # get line model
        line_pts = self.compute_line_endpoints(tl_line_idx)
        if len(line_pts) > 0:
            a, b = line_params_from_pts(list(reversed(line_pts[:2])), list(reversed(line_pts[2:])))
            # get tl record
            tl_line_rec = self.get_tl_rec(tl_line_idx_cmp)
            # assemble basic hypothesis
            vec_x = tl_line_rec.prior_sign_xoff.values
            vec_y = tl_line_rec.prior_sign_xoff.apply(lambda x: a * x + b).values
            return np.stack([vec_x, vec_y], axis=1)
        else:
            raise Exception('No line assigned!')

    def create_null_hypo_segmentation(self, tl_line_idx, tl_line_idx_cmp=None):
        # allow (tl_line_idx != tl_line_idx_cmp)
        # tl_line_idx --> choice of line detection (line model, segmentation, detections)
        # tl_line_idx_cmp --> choice of transliteration line (tl_line_rec)
        if tl_line_idx_cmp is None:
            tl_line_idx_cmp = tl_line_idx
        self.check_line_idx_in_bounds(tl_line_idx, tl_line_idx_cmp)

        # get line model
        line_pts = self.compute_line_endpoints(tl_line_idx)
        a, b = line_params_from_pts(list(reversed(line_pts[:2])), list(reversed(line_pts[2:])))
        # get tl record
        tl_line_rec = self.get_tl_rec(tl_line_idx_cmp)
        tl_line_len = tl_line_rec.prior_line_len.values[0]
        # get information from segment
        bbox = self.get_line_segmentation_bbox(tl_line_idx)
        line_xmin = bbox[0] - 32
        line_xmax = bbox[2] + 32
        det_line_len = line_xmax - line_xmin
        # assemble hypothesis
        vec_x = line_xmin + (tl_line_rec.prior_sign_xoff.values * det_line_len / tl_line_len)
        vec_y = (lambda x: a * x + b)(vec_x)
        return np.stack([vec_x, vec_y], axis=1)

    def create_null_hypo_alignments(self, tl_line_idx, tl_line_idx_cmp=None):
        # allow (tl_line_idx != tl_line_idx_cmp)
        # tl_line_idx --> choice of line detection (line model, segmentation, detections)
        # tl_line_idx_cmp --> choice of transliteration line (tl_line_rec)
        if tl_line_idx_cmp is None:
            tl_line_idx_cmp = tl_line_idx
        self.check_line_idx_in_bounds(tl_line_idx, tl_line_idx_cmp)

        # get hypothesis using tl_line_idx and tl_line_idx_cmp
        sign_hypos = self.create_null_hypo_segmentation(tl_line_idx, tl_line_idx_cmp)
        # get unique labels from transliteration using tl_line_idx_cmp
        tl_line_rec = self.get_tl_rec(tl_line_idx_cmp)

        # filter transliteration
        #tl_select = (tl_line_rec.status == 1)  # (tl_line_rec.lbl < 240)
        labels = tl_line_rec.lbl.values
        bbox_list = []
        for x, lbl in zip(sign_hypos, labels):
            sign_width = self.stats.get_sign_width(lbl, sign_width=1)
            # compute bbox from center, width and height
            sign_h = self.stats.tblSignHeight
            sign_w = (sign_width * self.stats.tblSignHeight)
            bbox_list.append([x[0] - sign_w/2., x[1] - sign_h/2., x[0] + sign_w/2., x[1] + sign_h/2.])
        return labels, sign_hypos, np.stack(bbox_list)

    def create_conditional_hypo_alignments(self, tl_line_idx, tl_line_idx_cmp=None, anno_df=None,
                                           min_dets_inline=2, ncompl_thresh=-1, smooth_y=True):
        # allow (tl_line_idx != tl_line_idx_cmp)
        # tl_line_idx --> choice of line detection (line model, segmentation, detections)
        # tl_line_idx_cmp --> choice of transliteration line (tl_line_rec)
        if tl_line_idx_cmp is None:
            tl_line_idx_cmp = tl_line_idx
        self.check_line_idx_in_bounds(tl_line_idx, tl_line_idx_cmp)

        if anno_df is None:
            raise NotImplementedError

        # get tl record (only visible detections? Yes!)
        tl_line_rec = self.get_tl_rec(tl_line_idx_cmp)
        # get sign detections and anno_df record
        # using anno_df here, because it provides additional columns (would be cleaner if merged with region_detections)
        region_det = self.get_region_detections(tl_line_idx)
        line_det_df = anno_df.loc[region_det[:, -1]]
        line_det = region_det

        # select detections that match line idx and that are reliable (good completeness)
        # there is a conflict because anno_df and region_detections might use different line tl alignment
        # It is important the same alignment method is used for creation of anno_df and this imputation
        # -> then this will be fine most of time
        select_line_idx = (line_det_df.line_idx == tl_line_idx_cmp)
        if ncompl_thresh > 0:
            select_line_idx = (line_det_df.ncompl > ncompl_thresh) & select_line_idx
        line_det_df = line_det_df[select_line_idx]
        line_det = line_det[select_line_idx, :]

        # only consider line if there are at least one aligned detection and one unaligned detection
        if (len(line_det_df) >= min_dets_inline) and (len(tl_line_rec) > len(line_det_df)):
            # get line model
            line_pts = self.compute_line_endpoints(tl_line_idx)
            a, b = line_params_from_pts(list(reversed(line_pts[:2])), list(reversed(line_pts[2:])))
            # get information from segment
            bbox = self.get_line_segmentation_bbox(tl_line_idx)
            line_xmin = bbox[0] - 32
            line_xmax = bbox[2] + 32
            section_xmin = line_xmin
            # assemble hypothesis
            list_sidx = []
            list_sign_hypos = []
            list_sign_boxes = []
            list_anno_df_idx = []
            # iterate over line transliteration
            for ii, (sidx, sign_rec) in enumerate(tl_line_rec.iterrows()):
                #print(sign_rec.pos_idx)
                # select detection at current position
                select_pos = (line_det_df.pos_idx == sign_rec.pos_idx)
                curr_det_rec = line_det_df[select_pos]
                curr_det = line_det[select_pos]
                if len(curr_det_rec) > 0:  # detection available at pos_idx?
                    if len(list_sidx) > 0:  # any unaligned signs available?
                        # update section xmax (left side of detection)
                        section_xmax = curr_det[:, 4]  # + 32 # + 0
                        # compute and append hypos
                        section_hypos = compute_section_hypo_vec(tl_line_rec, list_sidx, section_xmin, section_xmax, a, b)
                        list_sign_hypos.append(section_hypos)
                    # append detection
                    list_sign_hypos.append(curr_det[:, 1:3])
                    list_anno_df_idx.append(curr_det_rec.index.item())
                    # update section xmin (right side of detection)
                    section_xmin = curr_det[:, 6]  # - 32 # - 0
                    # reset sidx
                    list_sidx = []
                elif ii == len(tl_line_rec) - 1:  # last item in line
                    list_sidx.append(sidx)  # append last
                    # update section xmax (left end of line)
                    section_xmax = max(section_xmin + 128./2, line_xmax)
                    # compute and append hypos
                    section_hypos = compute_section_hypo_vec(tl_line_rec, list_sidx, section_xmin, section_xmax, a, b)
                    list_sign_hypos.append(section_hypos)
                    list_anno_df_idx.append(-1)
                else:
                    list_sidx.append(sidx)
                    list_anno_df_idx.append(-1)

            # to ndarray
            t_hypos = np.concatenate(list_sign_hypos)
            t_anno_idx = np.array(list_anno_df_idx)

            # encode distance in anno_vec
            vec_dt = distance_transform_edt(t_anno_idx < 0)
            t_anno_idx[vec_dt > 0] = -vec_dt[vec_dt > 0]

            # perform regression in order to refine y coordinates
            if smooth_y:
                # select aligned detections
                select_dets = (t_anno_idx > -1)
                # select inlier imputations (not marked)
                select_inliers = np.all(t_hypos > -1, axis=1)
                lr = LinearRegression()
                x, y = t_hypos[:, 0].reshape(-1, 1), t_hypos[:, 1]
                sample_weight = np.ones_like(y) * select_inliers
                sample_weight += select_dets  #  / 2.
                lr.fit(x, y, sample_weight=sample_weight)
                #print(lr.coef_, lr.intercept, a, b)
                # assign smoothed values
                t_hypos[~select_dets, 1] = lr.predict(x)[~select_dets]

            # iterate over line transliteration
            for ii, (sidx, sign_rec) in enumerate(tl_line_rec.iterrows()):
                # select detection at current position
                select_pos = (line_det_df.pos_idx == sign_rec.pos_idx)
                curr_det = line_det[select_pos]
                if len(curr_det) > 0:  # detection available at pos_idx?
                    list_sign_boxes.append(curr_det[:, 4:8])
                else:
                    x = t_hypos[ii, :]
                    sign_h = self.stats.tblSignHeight
                    sign_w = sign_rec.prior_sign_width
                    list_sign_boxes.append(np.array([[x[0] - sign_w / 2., x[1] - sign_h / 2.,
                                                     x[0] + sign_w / 2., x[1] + sign_h / 2.]]))
            # to ndarray
            t_boxes = np.concatenate(list_sign_boxes)
            return t_hypos, t_boxes, t_anno_idx, tl_line_rec[['lbl', 'line_idx', 'pos_idx']].values
        else:
            return [], [], [], []

    # score line-transliteration assignments / produce alignments

    def compute_weak_score(self, tl_line_idx, tl_line_idx_cmp=None):
        # for sign detections that are present in assigned transliteration, accumulate confidence
        # ignores geometry and requires no sign hypothesis!
        # provides a weak score for the line-transliteration assignment

        # allow (tl_line_idx != tl_line_idx_cmp)
        # tl_line_idx --> choice of line detection (line model, segmentation, detections)
        # tl_line_idx_cmp --> choice of transliteration line (tl_line_rec)
        if tl_line_idx_cmp is None:
            tl_line_idx_cmp = tl_line_idx
        self.check_line_idx_in_bounds(tl_line_idx, tl_line_idx_cmp)

        # get unique labels from transliteration using tl_line_idx_cmp
        uiq_labels = self.get_tl_rec(tl_line_idx_cmp).lbl.unique()
        # get detections using tl_line_idx
        region_det = self.get_region_detections(tl_line_idx, ortho_pad=self.dist_interline_median / 2.)  # 6. -> 2.
        det_df = pd.DataFrame(region_det, columns=['lbl', 'cx', 'cy', 'score', 'x1', 'y1', 'x2', 'y2', 'idx'])
        # get max scores per class
        max_conf_det = det_df.groupby('lbl').score.max()
        # intersect with unique
        intersect_det = max_conf_det[max_conf_det.index.isin(uiq_labels)]
        # catch if no matching sign present
        if len(intersect_det) > 0:
            return (len(intersect_det) / float(len(uiq_labels))) * intersect_det.mean()
        else:
            return 0

    def compute_bleu_score(self, tl_line_idx, tl_line_idx_cmp=None):
        # allow (tl_line_idx != tl_line_idx_cmp)
        # tl_line_idx --> choice of line detection (line model, segmentation, detections)
        # tl_line_idx_cmp --> choice of transliteration line (tl_line_rec)
        if tl_line_idx_cmp is None:
            tl_line_idx_cmp = tl_line_idx
        self.check_line_idx_in_bounds(tl_line_idx, tl_line_idx_cmp)

        # get unique labels from transliteration using tl_line_idx_cmp  (OMG, why would I want UNIQUE labels?!)
        uiq_labels = self.get_tl_rec(tl_line_idx_cmp).lbl.values  # unique()
        # get detections using tl_line_idx
        region_det = self.get_region_detections(tl_line_idx, ortho_pad=self.dist_interline_median / 2.)  # 6. -> 2.
        det_df = pd.DataFrame(region_det, columns=['lbl', 'cx', 'cy', 'score', 'x1', 'y1', 'x2', 'y2', 'idx'])
        # get max scores per class
        max_conf_det = det_df.groupby('lbl').score.max()

        # sort detections according to cx
        det_df = det_df.sort_values('cx')
        # filter low score detections
        det_df = det_df[det_df.score > 0.5]  # 0.5
        reference = uiq_labels
        candidate = det_df.lbl.values
        if 1:
            # compute score
            score = compute_bleu(candidate, reference)
            return 1 - score
        else:
            # compute edit distance score
            score = compute_levenshtein(candidate, reference)
            return score

    def compute_ransac_score(self, tl_line_idx, tl_line_idx_cmp=None, max_dist_thresh=2, dist_weight=1,
                             normalization_factor=128/2., return_alignments=False):
        # for each sign in sign hypothesis, check distance to next matching detection and the confidence
        # incorporates geometry and require sign hypothesis!
        # provides a basic score for the sign-hypothesis
        # provides an implicit score for line-transliteration assignment

        # allow (tl_line_idx != tl_line_idx_cmp)
        # tl_line_idx --> choice of line detection (line model, segmentation, detections)
        # tl_line_idx_cmp --> choice of transliteration line (tl_line_rec)
        if tl_line_idx_cmp is None:
            tl_line_idx_cmp = tl_line_idx
        self.check_line_idx_in_bounds(tl_line_idx, tl_line_idx_cmp)

        # get unique labels from transliteration using tl_line_idx_cmp
        tl_line_rec = self.get_tl_rec(tl_line_idx_cmp)
        # get detections using tl_line_idx
        region_det = self.get_region_detections(tl_line_idx, ortho_pad=self.dist_interline_median / 2.)  # 6. TD changed to 2. because otherwise many good detections lost
        # get hypothesis using tl_line_idx and tl_line_idx_cmp
        sign_hypos = self.create_null_hypo_segmentation(tl_line_idx, tl_line_idx_cmp)
        # sign_hypos = self.create_null_hypo_basic(tl_line_idx, tl_line_idx_cmp)

        # evaluate hypothesis
        norm_score, alignments = self._ransac_score(tl_line_rec, region_det, sign_hypos,
                                                    max_dist_thresh, dist_weight, normalization_factor)
        if return_alignments:
            return norm_score, alignments, region_det, sign_hypos, tl_line_rec
        else:
            return norm_score

    def _ransac_score(self, tl_line_rec, region_det, sign_hypos, max_dist_thresh, dist_weight, normalization_factor):
        # could be put outside class, but for what reason?

        # code adopted from Hypothesis class
        num_signs = len(tl_line_rec)
        max_sing_score = 1 + max_dist_thresh * dist_weight  # max_confidence + max_dist * dist_weight
        max_score = max_sing_score * num_signs

        # start with a score of zero
        score = 0
        alignments = np.ones((num_signs, 1), dtype=int) * (-1)

        if len(region_det) > 0:
            # for each sign in transliteration
            for i in range(num_signs):

                min_dist_score = max_sing_score
                bbox_idx = 0
                x = sign_hypos[i, 0:2]

                label = tl_line_rec.lbl.iloc[i]
                right_label = (label != 0)

                # make sure sign exists
                if right_label:
                    if False:
                        if label in self.stats.signs.keys():
                            sign_width = self.stats.signs[label]["median"]["width"]
                        else:
                            sign_width = 1
                    else:
                        sign_width = self.stats.get_sign_width(label, sign_width=1)

                    # for each detection point with corresponding label
                    det_same_label_idx = np.random.permutation(np.nonzero(region_det[:, 0] == label)[0])

                    # compute pairwise distances
                    var = np.array([sign_width * 1, 1], dtype=np.float)
                    nXY = cdist(np.array([x]), region_det[det_same_label_idx, 1:3], metric='seuclidean',
                                V=var).squeeze() / normalization_factor
                    # nXY = cdist(np.array([x]), region_det[det_same_label_idx, 1:3], metric='euclidean'
                    #             ).squeeze() / normalization_factor

                    # compute scores
                    dist = 1 - region_det[det_same_label_idx, 3] + nXY * dist_weight
                    # if any dets in range, get best and save its score and alignment
                    in_range = np.where(nXY <= max_dist_thresh)
                    in_range = in_range[0]
                    if len(in_range) > 0:
                        good_dist = dist[in_range]
                        good_dets = det_same_label_idx[in_range]
                        idx = np.argmin(good_dist)
                        min_dist_score = good_dist[idx]
                        bbox_idx = good_dets[idx]
                        alignments[i] = np.int(bbox_idx)

                score += min_dist_score

        return score / max_score, alignments  # score, score / max_score, aligned

    def compute_line_matching_score(self, tl_line_idx, tl_line_idx_cmp=None, return_alignments=False, param_dict=None):
        # formulate line matching as energy minimization problem that yields a sign hypothesis
        # incorporates geometry and require NO sign hypothesis!
        # provides an implicit score for line-transliteration assignment
        # allow (tl_line_idx != tl_line_idx_cmp)
        # tl_line_idx --> choice of line detection (line model, segmentation, detections)
        # tl_line_idx_cmp --> choice of transliteration line (tl_line_rec)
        if tl_line_idx_cmp is None:
            tl_line_idx_cmp = tl_line_idx
        self.check_line_idx_in_bounds(tl_line_idx, tl_line_idx_cmp)
        #print(tl_line_idx, tl_line_idx_cmp)

        # get unique labels from transliteration using tl_line_idx_cmp
        tl_line_rec = self.get_tl_rec(tl_line_idx_cmp)
        # get detections using tl_line_idx
        region_det = self.get_region_detections(tl_line_idx, ortho_pad=self.dist_interline_median / 2.)  # 3.
        # get line endpoints using tl_line_idx
        line_pts = self.compute_line_endpoints(tl_line_idx)
        line_pts_arr = np.fliplr(np.array(line_pts).reshape((2, 2)))  # pretty format
        line_rec = self.get_line_rec(tl_line_idx)
        # get hypothesis using tl_line_idx and tl_line_idx_cmp
        sign_hypos = self.create_null_hypo_segmentation(tl_line_idx, tl_line_idx_cmp)
        # create line matching problem (make sure that tl_line_rec a deep copy! -> otherwise pandas warnings)
        line_gm = LineMatching1D(tl_line_rec.copy(deep=True), region_det, line_rec, line_pts_arr,
                                 self.stats, sign_hypos=sign_hypos, param_dict=param_dict)
        # run inference
        try:
            line_gm.run_inference()
        except KeyError:
            print('error for', tl_line_idx, tl_line_idx_cmp)
            raise

        # check if anything to match and assign energy accordingly
        if line_gm.num_relevant > 0:
            norm_score = line_gm.energy  # line_gm.energy  #
        else:
            norm_score = 1.0   # cost normalized to [0, 1] interval #line_gm.max_cost

        if return_alignments:
            #alignments = line_gm.get_region_alignments()
            sign_hypos = self.create_null_hypo_segmentation(tl_line_idx, tl_line_idx_cmp)

            return norm_score, region_det, sign_hypos, line_gm.tl_line_rec
        else:
            return norm_score

    # full tablet functions

    def tab_compute_line_matching_score(self, tl_line_idx_list=None, tl_line_idx_cmp_list=None, param_dict=None):
        # tl_line_idx_list --> choice of line detection (line model, segmentation, detections)
        # tl_line_idx_cmp_list --> choice of transliteration line (tl_line_rec)
        if tl_line_idx_list is None:
            assigned_tl_indices = self.get_assigned_lines_idx()
            tl_line_idx_list = assigned_tl_indices
            tl_line_idx_cmp_list = assigned_tl_indices
        else:
            assert len(tl_line_idx_cmp_list) == len(tl_line_idx_list)

        list_sign_hypos = []
        list_tl_df = []
        # collect alignments from all lines
        for idx, idx_cmp in zip(tl_line_idx_list, tl_line_idx_cmp_list):
            (_, region_det, sign_hypos,
             gm_tl_line_rec) = self.compute_line_matching_score(idx, idx_cmp, return_alignments=True,
                                                                param_dict=param_dict)
            # append
            list_sign_hypos.append(sign_hypos)
            list_tl_df.append(gm_tl_line_rec)

        if len(list_tl_df) > 0:
            # merge values together
            tablet_tl_df = pd.concat(list_tl_df)
            sign_hypos = np.vstack(list_sign_hypos)
        else:
            tablet_tl_df = pd.DataFrame()
            sign_hypos = []

        # return
        return sign_hypos, tablet_tl_df

    def tab_get_gm_alignments(self, tl_line_idx_list=None, tl_line_idx_cmp_list=None, refined=False, min_dist=128.,
                              param_dict=None):
        # tl_line_idx_list --> choice of line detection (line model, segmentation, detections)
        # tl_line_idx_cmp_list --> choice of transliteration line (tl_line_rec)
        if tl_line_idx_list is None:
            assigned_tl_indices = self.get_assigned_lines_idx()
            tl_line_idx_list = assigned_tl_indices
            tl_line_idx_cmp_list = assigned_tl_indices

        (sign_hypos, tablet_tl_df) = self.tab_compute_line_matching_score(tl_line_idx_list, tl_line_idx_cmp_list,
                                                                          param_dict=param_dict)

        # refine and convert to eval format
        (aligned_list, aligned_bbox, tablet_tl_df_sel) =\
            self.prepare_aligned_detections(tablet_tl_df, refined=refined, min_dist=min_dist)

        return aligned_list, tablet_tl_df

    def tab_visualize_gm_alignments(self, tl_line_idx_list=None, tl_line_idx_cmp_list=None, ax=None,
                                    show_bbox=False, show_null_hypo=True, refined=False, min_dist=80.):    # min_dist=128.
        # tl_line_idx_list --> choice of line detection (line model, segmentation, detections)
        # tl_line_idx_cmp_list --> choice of transliteration line (tl_line_rec)
        if tl_line_idx_list is None:
            assigned_tl_indices = self.get_assigned_lines_idx()
            tl_line_idx_list = assigned_tl_indices
            tl_line_idx_cmp_list = assigned_tl_indices

        (sign_hypos, tablet_tl_df) = self.tab_compute_line_matching_score(tl_line_idx_list, tl_line_idx_cmp_list)

        # refine and convert to eval format
        (aligned_list, aligned_bbox, tablet_tl_df_sel) =\
            self.prepare_aligned_detections(tablet_tl_df, refined=refined, min_dist=min_dist)

        # plot
        if ax is None:
            fig, ax = plt.subplots(figsize=(15, 12))

        if show_null_hypo:
            # plot null hypo coordinates
            ax.plot(sign_hypos[:, 0], sign_hypos[:, 1], 'd', color=self.box_cmap(0.25), markersize=9, label='null hypo')

        # check if there are any alignments for the segment
        if 'region_det_idx' in tablet_tl_df.columns:   # could also use: if len(aligned_list) > 0

            if show_null_hypo:
                # plot null hypo lines
                for line_idx in tablet_tl_df.line_idx.unique():
                    line_select = (tablet_tl_df.line_idx == line_idx).values
                    ax.plot(sign_hypos[line_select, 0], sign_hypos[line_select, 1], color=self.box_cmap(0.25), linewidth=1.5)

            # # avoid outliers in tablet_tl_df
            # no_outlier_select = (tablet_tl_df.region_det_idx >= 0)
            # tablet_tl_df_sel = tablet_tl_df[no_outlier_select]

            # plot aligned detections
            global_alignments = tablet_tl_df_sel.aligned_det_idx.values.astype(int)
            # aligned = self.detections[global_alignments, 1:3]
            aligned = compute_bbox_ctr(aligned_bbox)  # use refined

            ax.plot(aligned[:, 0], aligned[:, 1], 's', color=self.box_cmap(0.6), markersize=10, label='aligned detections')

            # connect aligned detections
            for line_idx in tablet_tl_df_sel.line_idx.unique():
                line_select = (tablet_tl_df_sel.line_idx == line_idx).values
                ax.plot(aligned[line_select, 0], aligned[line_select, 1], color=self.box_cmap(0.6), linewidth=2.0)

            if show_bbox:
                # aligned_bbox = self.detections[global_alignments, 4:8]
                # aligned_bbox = self.tab_refine_detections_using_line_hypos(self.detections[global_alignments, :], tablet_tl_df_sel, min_dist=min_dist, use_segmentation=True)

                cnn_confidence = self.detections[global_alignments, 3]
                gm_confidence = tablet_tl_df_sel.nE.values

                plot_boxes(aligned_bbox, confidence=cnn_confidence, ax=ax)

            # annotate aligned detections
            for i, pos_idx in enumerate(tablet_tl_df_sel.pos_idx.values):
                ax.annotate(pos_idx, (aligned[i, 0], aligned[i, 1]), fontsize=10, weight='bold',
                            color=plt.cm.gray(0.05), va='center', ha='center')

        # plot tablet
        ax.imshow(self.input_im, cmap='gray')
        ax.legend(fancybox=True, loc='best')
        ax.axis('off')

        return aligned_list, tablet_tl_df

    # evaluate alignments

    def prepare_aligned_detections(self, tablet_tl_df, refined=False, min_dist=128.):
        # create empty containers
        aligned_list, aligned_bbox = [], []

        # check if there are any alignments for the segment
        if 'region_det_idx' in tablet_tl_df.columns:

            # filter tablet_tl_df
            no_outlier_select = (tablet_tl_df.aligned_det_idx >= 0)  # avoid outliers in tablet_tl_df [outlier -1]
            not_nan_select = pd.notna(tablet_tl_df.aligned_det_idx)  # avoid NANs, if inference fails
            tablet_tl_df_sel = tablet_tl_df[no_outlier_select & not_nan_select]
            # get aligned detections
            global_alignments = tablet_tl_df_sel.aligned_det_idx.values.astype(int)
            aligned_det = self.detections[global_alignments, :].copy()
            aligned_bbox = aligned_det[:, 4:8]

            if refined and len(aligned_det) > 0:
                # refine bboxes and update aligned detections
                aligned_bbox = self.tab_refine_detections_using_line_hypos(aligned_det, tablet_tl_df_sel,
                                                                           min_dist=min_dist, use_segmentation=True)
                aligned_det[:, 4:8] = aligned_bbox
            # convert to list of list format
            aligned_list = convert_alignments_for_eval(aligned_det)
        else:
            tablet_tl_df_sel = tablet_tl_df

        return aligned_list, aligned_bbox, tablet_tl_df_sel

    def tab_create_null_hypo_alignments(self, tl_line_idx_list=None, tl_line_idx_cmp_list=None):
        # tl_line_idx_list --> choice of line detection (line model, segmentation, detections)
        # tl_line_idx_cmp_list --> choice of transliteration line (tl_line_rec)
        if tl_line_idx_list is None:
            assigned_tl_indices = self.get_assigned_lines_idx()
            tl_line_idx_list = assigned_tl_indices
            tl_line_idx_cmp_list = assigned_tl_indices

        det_bbox_list = []
        for tl_line_idx, tl_line_idx_cmp in zip(tl_line_idx_list, tl_line_idx_cmp_list):
            l_lbls, l_boxctrs, l_boxes = self.create_null_hypo_alignments(tl_line_idx, tl_line_idx_cmp)
            # collect [ID, cx, cy, score, x1, y1, x2, y2, idx]
            det_bbox_list.append(np.hstack((l_lbls.reshape(-1, 1), l_boxctrs,
                                            np.ones_like(l_lbls).reshape(-1, 1), l_boxes)))
        # stack and return
        return np.vstack(det_bbox_list)

    def tab_create_conditional_hypo_alignments(self, tl_line_idx_list=None, tl_line_idx_cmp_list=None, anno_df=None,
                                               min_dets_inline=2, ncompl_thresh=10, smooth_y=True, max_dist_det=0):
        # tl_line_idx_list --> choice of line detection (line model, segmentation, detections)
        # tl_line_idx_cmp_list --> choice of transliteration line (tl_line_rec)
        if tl_line_idx_list is None:
            assigned_tl_indices = self.get_assigned_lines_idx()
            tl_line_idx_list = assigned_tl_indices
            tl_line_idx_cmp_list = assigned_tl_indices

        det_bbox_list = []
        t_anno_idx_list = []
        t_meta_list = []
        for tl_line_idx, tl_line_idx_cmp in zip(tl_line_idx_list, tl_line_idx_cmp_list):
            (t_hypos, t_boxes, t_anno_idx,
             t_meta) = self.create_conditional_hypo_alignments(tl_line_idx, tl_line_idx_cmp, anno_df,
                                                               min_dets_inline, ncompl_thresh, smooth_y)
            if len(t_hypos) > 0:  # check if any alignment
                # filter non overlapping (negative x coordinate is marker)
                select_hypos = np.all(t_hypos > -1, axis=1)
                # filter using max_dist_det
                if max_dist_det > 0:
                    select_hypos &= (-max_dist_det <= t_anno_idx) & (t_anno_idx < 0)
                # apply filter
                t_hypos = t_hypos[select_hypos, :]
                t_boxes = t_boxes[select_hypos, :]
                t_anno_idx = t_anno_idx[select_hypos]
                t_meta = t_meta[select_hypos, :]
                # lbl vector
                t_lbls = t_meta[:, 0].reshape(-1, 1)
                # collect [ID, cx, cy, score, x1, y1, x2, y2, idx]
                det_bbox_list.append(np.hstack((t_lbls, t_hypos, np.ones_like(t_lbls), t_boxes)))
                t_anno_idx_list.append(t_anno_idx)
                t_meta_list.append(t_meta)
        if len(det_bbox_list) > 0:
            # stack and return
            return np.vstack(det_bbox_list), np.hstack(t_anno_idx_list), np.vstack(t_meta_list)
        else:
            return [], [], []

    def tab_compute_accuracy(self, gt_boxes, gt_labels, tl_line_idx_list=None, tl_line_idx_cmp_list=None,
                             refined=False, min_dist=128.):
        # tl_line_idx_list --> choice of line detection (line model, segmentation, detections)
        # tl_line_idx_cmp_list --> choice of transliteration line (tl_line_rec)
        if tl_line_idx_list is None:
            assigned_tl_indices = self.get_assigned_lines_idx()
            tl_line_idx_list = assigned_tl_indices
            tl_line_idx_cmp_list = assigned_tl_indices

        # only run if gt available
        if len(gt_boxes) > 0:
            # compute gm alignments
            sign_hypos, tablet_tl_df = self.tab_compute_line_matching_score(tl_line_idx_list, tl_line_idx_cmp_list)
            # convert to eval format
            aligned_list, _, _ = self.prepare_aligned_detections(tablet_tl_df, refined=refined, min_dist=min_dist)
            # get accuracy
            acc = compute_accuracy(gt_boxes, gt_labels, aligned_list)
            return acc
        else:
            return -1

    # refine across tablet

    def tab_refine_detections_using_line_hypos(self, detections_sel, tablet_tl_df_sel, min_dist=128.,
                                               use_segmentation=False):
        # basic refinement only uses linear line model, however, line detection is more accurate than that!
        # line segmentation mask is used to find actual line center!

        det_bbox_list = []
        for tl_line_idx in tablet_tl_df_sel.line_idx.unique():
            line_select = (tablet_tl_df_sel.line_idx == tl_line_idx).values
            region_detections = detections_sel[line_select, :]
            det_boxes = self.refine_detections_using_line_hypos(tl_line_idx, region_detections, min_dist=min_dist,
                                                                use_segmentation=use_segmentation)
            # collect
            det_bbox_list.append(det_boxes)
        # stack and return
        return np.vstack(det_bbox_list)

    # generate training data

    def tab_generate_training_data(self, folder, destination, image_name,
                                   image_path, scale, seg_idx,
                                   seg_bbox, tablet_tl_df, label_list, append=True):
        # check if there are any alignments for the segment
        if 'region_det_idx' in tablet_tl_df.columns:

            (aligned_list, aligned_bbox, tablet_tl_df_sel) =\
                self.prepare_aligned_detections(tablet_tl_df, refined=True, min_dist=80.)

            align_ratio = len(tablet_tl_df_sel)/float(len(tablet_tl_df))

            # get aligned detections
            global_alignments = tablet_tl_df_sel.aligned_det_idx.values.astype(int)
            #aligned_det = self.detections[global_alignments, :].copy()
            #aligned_bbox = aligned_det[:, 4:8]

            if append:
                file_attribute = "a"
            else:
                file_attribute = "w"

            with open(destination, file_attribute) as examples_file:
                if not append:
                    examples_file.write("imageName, folder, image_path, label, train_label, x1, y1, x2, y2, "
                                        "width, height, segm_idx, line_number, sign_position_in_line, "
                                        "score, m_score, align_ratio\n")

                for idx, al in enumerate(global_alignments):
                    #if al[0] != 0:

                    line_rec = tablet_tl_df_sel.iloc[idx]
                    label_new = line_rec.lbl

                    detection = self.detections[al].copy()
                    label_new = int(detection[0])
                    score = detection[3]

                    bbox = np.around(aligned_bbox[idx] / scale, 1)  # re-scale bboxes to original image size
                    width = bbox[2] - bbox[0] + 1
                    height = bbox[3] - bbox[1] + 1

                    if len(seg_bbox) > 0:
                        # map detection bbox to global coordinate system by translation
                        bbox = convert_bbox_local2global(bbox, list(seg_bbox))
                    # write to file
                    examples_file.write("{img}, {fold}, {impath}, {lab}, {newLab:d}, {x1}, {y1}, {x2}, {y2},"
                                        " {w}, {h}, {segm_idx}, {l:d}, {spos:d}, "
                                        "{score}, {m_score:.4}, {align}\n"
                                        .format(img=image_name, fold=folder, impath=image_path,
                                                lab=label_list.index(label_new), newLab=label_new,
                                                x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3],
                                                w=width, h=height, segm_idx=seg_idx,
                                                l=int(line_rec.line_idx), spos=int(line_rec.pos_idx),
                                                score=score, m_score=line_rec.nE, align=align_ratio))

    def tab_generate_null_hypo_training_data(self, folder, destination, image_name, image_path, scale,
                                             seg_idx, seg_bbox, label_list, append=True):
        # create null hypo alignments
        sign_hypo_detections = self.tab_create_null_hypo_alignments()

        if append:
            file_attribute = "a"
        else:
            file_attribute = "w"

        with open(destination, file_attribute) as examples_file:
            if not append:
                examples_file.write("imageName, folder, image_path, label, train_label, x1, y1, x2, y2, "
                                    "width, height, segm_idx, line_number, sign_position_in_line, "
                                    "score, m_score, align_ratio\n")

            for idx, detection in enumerate(sign_hypo_detections):
                label_new = int(detection[0])
                score = detection[3]

                bbox = np.around(detection[4:8] / scale, 1)  # re-scale bboxes to original image size
                width = bbox[2] - bbox[0] + 1
                height = bbox[3] - bbox[1] + 1

                if len(seg_bbox) > 0:
                    # map detection bbox to global coordinate system by translation
                    bbox = convert_bbox_local2global(bbox, list(seg_bbox))
                # write to file
                examples_file.write("{img}, {fold}, {impath}, {lab}, {newLab:d}, {x1}, {y1}, {x2}, {y2},"
                                    " {w}, {h}, {segm_idx}, {l:d}, {spos:d}, "
                                    "{score}, {m_score:.4}, {align}\n"
                                    .format(img=image_name, fold=folder, impath=image_path,
                                            lab=label_list.index(label_new), newLab=label_new,
                                            x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3],
                                            w=width, h=height, segm_idx=seg_idx,
                                            l=int(1), spos=int(1),
                                            score=score, m_score=1., align=1.))

    def tab_generate_cond_hypo_training_data(self, folder, destination, image_name, image_path, scale,
                                             seg_idx, seg_bbox, label_list, append=True, anno_df=None,
                                             min_dets_inline=2, ncompl_thresh=-1, smooth_y=True, max_dist_det=3):
        # create null hypo alignments
        (sign_hypo_detections, tab_anno_idx,
         tab_meta) = self.tab_create_conditional_hypo_alignments(anno_df=anno_df, min_dets_inline=min_dets_inline,
                                                                 ncompl_thresh=ncompl_thresh, smooth_y=smooth_y,
                                                                 max_dist_det=max_dist_det)

        if append:
            file_attribute = "a"
        else:
            file_attribute = "w"

        with open(destination, file_attribute) as examples_file:
            if not append:
                examples_file.write("imageName, folder, image_path, label, train_label, x1, y1, x2, y2, "
                                    "width, height, segm_idx, line_number, sign_position_in_line, "
                                    "score, m_score, align_ratio\n")

            for idx, detection in enumerate(sign_hypo_detections):
                label_new = int(detection[0])
                score = detection[3]
                l_idx = tab_meta[idx, 1]
                pos_idx = tab_meta[idx, 2]

                # compute align ratio
                select_seg_idx = anno_df.seg_idx == seg_idx
                if np.any(select_seg_idx):
                    align_ratio = anno_df[select_seg_idx].align_ratio.head(1).item()
                else:
                    align_ratio = 0

                bbox = np.around(detection[4:8] / scale, 1)  # re-scale bboxes to original image size
                width = bbox[2] - bbox[0] + 1
                height = bbox[3] - bbox[1] + 1

                if len(seg_bbox) > 0:
                    # map detection bbox to global coordinate system by translation
                    bbox = convert_bbox_local2global(bbox, list(seg_bbox))
                # write to file
                examples_file.write("{img}, {fold}, {impath}, {lab}, {newLab:d}, {x1}, {y1}, {x2}, {y2},"
                                    " {w}, {h}, {segm_idx}, {l:d}, {spos:d}, "
                                    "{score}, {m_score:.4}, {align}\n"
                                    .format(img=image_name, fold=folder, impath=image_path,
                                            lab=label_list.index(label_new), newLab=label_new,
                                            x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3],
                                            w=width, h=height, segm_idx=seg_idx,
                                            l=int(l_idx), spos=int(pos_idx),
                                            score=score, m_score=1., align=align_ratio))

    # test case: no transliteration available!
    # 1) line detection is used for selection of detections
    # 2) line-wise nms
    # 3) detection boxes are refined using line segmentation masks

    def tab_nms_and_refine_detections_using_line_hypos(self, nms_thres=0.4, ortho_pad=None):
        # returns detection array of the form:
        # [ID, cx, cy, score, x1, y1, x2, y2, idx]

        if ortho_pad is None:
            # ortho_pad is important parameter
            # better choose pad smaller than self.dist_interline_median/2.0
            ortho_pad = self.dist_interline_median/3.0

        # for each line perform nms followed by bbox refinement
        filtered_dets = []
        for line_idx in self.tl_df.line_idx.values:

            region_dets = self.get_region_detections(line_idx, ortho_pad=ortho_pad)

            # if there are any region detections, apply nms, refine and collect
            if len(region_dets) > 0:
                # 1) nms (optional)
                # nms_thres = 0.4
                if True:  # True
                    keep_inds = nms(region_dets[:, list(range(4, 8)) + [3]], nms_thres)
                    region_dets = region_dets[keep_inds]

                # 2) refine boxes
                det_boxes = self.refine_detections_using_line_hypos(line_idx, region_dets, use_segmentation=True,
                                                                    min_dist=80.)   # 128/2. different min_dist usually hurts
                # assemble new detection array
                refined_region_dets = update_detections_array(region_dets, det_boxes)

                # 3) nms (optional)
                # nms_thres = 0.4
                if False:
                    keep_inds = nms(refined_region_dets[:, list(range(4, 8)) + [3]], nms_thres)
                    refined_region_dets = refined_region_dets[keep_inds]

                # store in list
                filtered_dets.append(refined_region_dets)

        # stack result to ndarray
        filtered_dets = np.vstack(filtered_dets)
        return filtered_dets
