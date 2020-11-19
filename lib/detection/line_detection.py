import numpy as np
import pandas as pd

import torch
from torchvision import transforms as trafos

from skimage import draw
from skimage.color import label2rgb
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
from skimage.morphology import skeletonize, skeletonize_3d, thin, medial_axis, watershed

from scipy import ndimage as ndi
from scipy.spatial.distance import pdist, cdist, squareform

from ..detection.detection_helpers import label_map2image, coord_in_image


# prepare input for line detection

def preprocess_line_input(pil_im, scale, shift=None):
    """ produces five copies of the segment at slightly different offsets

    :param pil_im: tablet segment that is to be processed
    :param scale: scale which should be used for resizing
    :param shift: offset shift used to produce five-fold oversampling
    :return: 4D tensor with 5xCxWxH
    """
    if shift is None:
        shift = 0  # cfg.TEST.SHIFT
    # compute scaled size
    imw, imh = pil_im.size
    imw = int(imw * scale)
    imh = int(imh * scale)
    # determine crop size
    crop_sz = [int(imw - shift), int(imh - shift)]
    # tensor-space transforms
    ts_transform = trafos.Compose([
        trafos.ToTensor(),
        trafos.Normalize(mean=[0.5], std=[1]), # normalize
    ])
    # compose transforms
    tablet_transform = trafos.Compose([
            trafos.Lambda(lambda x: x.convert('L')), # convert to gray
            trafos.Resize((imh, imw)), # resize according to scale
            trafos.FiveCrop((crop_sz[1], crop_sz[0])), # oversample
            trafos.Lambda(
                lambda crops: torch.stack([ts_transform(crop) for crop in crops])), # returns a 4D tensor
        ])
    # apply transforms
    im_list = tablet_transform(pil_im)
    return im_list


def apply_detector(inputs, model_fcn, device):

    with torch.no_grad():  # faster, less memory usage
        inputs = inputs.to(device)
        # apply network
        output = model_fcn(inputs)
        # convert to numpy
        output = output.data.cpu().numpy()
        return output


# prepare transliteration for line detection

def prepare_transliteration(tl_df, num_lines, stats):
    """
    ATTENTION: this filters the transliteration according to status!
    """

    # prepare transliteration for line detection
    if num_lines > 0:
        # only visible/not broken
        tl_df = tl_df[tl_df.status > 0]
        # compute line length
        tl_df = tl_df.groupby('line_idx').apply(compute_line_length_from_tl, stats)
        # get line statistics
        num_vis_lines = tl_df.line_idx.nunique()  # num visible lines (not broken lines)
        # len_lines = tl_df.groupby('line_idx').pos_idx.count()
        len_lines = tl_df.line_idx.value_counts()
        len_min, len_max = len_lines.min(), len_lines.max()
    else:
        # TODO: if no tl info available, use initial line detection results to set these parameters
        len_min, len_max = 4, 12
        num_vis_lines = 40

    return tl_df, num_vis_lines, len_min, len_max


# extract lines with hough transform

def compute_hough_transform(line_det_map1, line_det_map2, re_focus_angle=True):
    # focus theta for cuneiform horizontal lines
    theta_range = np.linspace(np.deg2rad(83), np.deg2rad(97), 50)
    # theta_range = np.linspace(np.deg2rad(-90) ,np.deg2rad(90), 180) # normal range

    # Classic straight-line Hough transform (usually angles from -90 to +90)
    h, theta, d = hough_line(line_det_map1, theta=theta_range)

    # debug
    # plt.imshow(np.log(1 + h), extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]], cmap='gray', aspect=1/1.5)
    # plt.show()

    # focus angle and re-run
    if re_focus_angle:
        # get peaks
        accum, angles, dists = hough_line_peaks(h, theta, d, min_distance=1, min_angle=16, num_peaks=50)

        # get median angle
        m_angle = np.median(np.rad2deg(angles))
        # modify theta
        theta_range = np.linspace(np.deg2rad(m_angle - 2), np.deg2rad(m_angle + 2), 50)
        theta_range2 = np.linspace(np.deg2rad(m_angle - 3), np.deg2rad(m_angle + 3), 50)
        # Classic straight-line Hough transform (usually angles from -90 to +90)
        h, theta, d = hough_line(line_det_map2, theta=theta_range)

    return h, theta, d, theta_range, theta_range2


# group lines together that are "close"

def shoelace_formula(points):
    ''' compute are of polygon according to shoelace
    requires ordering of point coordinates
    https://en.wikipedia.org/wiki/Shoelace_formula

    :param points: 2xn matrix, where n is number of points (points need to be ordered!!)
    :return: area of polygon
    '''
    area = 0
    dmat = np.ones((2, 2))
    for i in range(points.shape[1]):
        dmat[:, 0] = points[:, i]
        dmat[:, 1] = points[:, (i+1) % points.shape[1]]
        area += np.linalg.det(dmat.transpose())
    return np.abs(area) / 2.


def area_between_two_line_segments(spt1, spt2, lpt1, lpt2):
    # compute area between line segments
    # assume: line segments do not intersect and
    # assume: pts should be order according to x-axis
    # this means a valid order would be [spt1, spt2, lpt2, lpt1]
    return shoelace_formula(np.stack([spt1, spt2, lpt2, lpt1], axis=1))


def nearby_and_near_parallel_2(l1, l2, interline_distance, interval=[0, 10]):
    # compute area between line segments over interval
    angle1, rad1 = l1
    angle2, rad2 = l2
    spt1, spt2 = line_pts_from_polar_line(angle1, rad1, x0=interval[0], x1=interval[1])
    lpt1, lpt2 = line_pts_from_polar_line(angle2, rad2, x0=interval[0], x1=interval[1])
    # use shoelace method
    area = area_between_two_line_segments(spt1, spt2, lpt1, lpt2)
    # check threshold
    interval_interline_area = interline_distance * np.abs(interval[1] - interval[0])

    # print area, interval_interline_area / 2.
    if area < interval_interline_area / 2.:
        return True
    else:
        return False


def nearby_and_near_parallel(l1, l2, interline_distance):
    # simple filter
    angle1, rad1 = l1
    angle2, rad2 = l2
    if np.abs(rad1 - rad2) < interline_distance/2. and np.abs(np.rad2deg(angle1-angle2)) < 1.0:
        return True
    else:
        return False


def do_intersect_in_interval(l1, l2, interval):
    # y = mx+c or in parametric form
    # \rho = x \cos \theta + y \sin \theta
    # \rho (radius) perpendicular distance from origin to the line
    # \theta is the angle formed by this perpendicular line

    angle1, rad1 = l1
    angle2, rad2 = l2
    lower, upper = interval

    quotient = (np.cos(angle1) - np.cos(angle2))

    if quotient == 0:  # same angles
        if np.abs(rad1 - rad2) < 3:  # same radius
            return True
        else:
            return False
    else:
        # compute intersection coordinate
        x_intersect = (rad1 - rad2) / quotient

        # inside interval
        if (x_intersect >= lower) and (x_intersect <= upper):
            return True
        else:
            return False


def compute_group_labels_from_dists(X_dist):
    # assign labels to groups
    # iterate over pairwise distances and

    # get squareform
    XX = squareform(X_dist)
    # set dummy labels
    labels = -np.ones(XX.shape[0])
    # label lines while checking for neighbourhood
    for ii in range(len(labels)):
        if labels[ii] == -1:
            labels[ii] = ii
        # for each row in squareform indicates potential neighbors
        for idx in np.where(XX[ii, :] > 0)[0]:
            labels[idx] = labels[ii]
    return labels


# associate lines with line segments


def line_pts_from_polar_line(angle, dist, x0=0, x1=10):
    # computes two points defining a line from polar line representation
    x0, x1 = x0 * np.ones_like(angle), x1 * np.ones_like(angle)  # x0 = np.zeros_like(angle)
    y0 = (dist - x0 * np.cos(angle)) / np.sin(angle)
    y1 = (dist - x1 * np.cos(angle)) / np.sin(angle)
    return (x0, y0), (x1, y1)


def line_params_from_pts(lpt1, lpt2):
    # compute parameters a, b for line representation y = a * x + b
    a = (lpt2[1] - lpt1[1])/float(lpt2[0] - lpt1[0])
    b = lpt1[1] - lpt1[0] * a
    return a, b


def normal_form_from_pts(p, q):
    # takes two points an computes
    # https://de.wikipedia.org/wiki/Normalenform#Aus_der_Zweipunkteform

    # normal
    n = np.array([-(q[1]-p[1]),
                  q[0]-p[0]], dtype=float)
    # normalize
    n_0 = n / np.linalg.norm(n)
    # distance from origin
    dist = np.dot(n_0, p)
    return n_0, dist


def hess_normal_form_from_pts(p, q):
    n_0, dist = normal_form_from_pts(p, q)
    # angle in rad
    rad = np.arctan(n_0[1]/n_0[0])
    return rad, dist


def _offset_pt_to_normal_form_line(pt, n_0, dist):
    return np.dot(n_0, pt) - dist


def _shift_pt_to_normal_form_line(pt, n_0, shift_dist):
    return pt + n_0 * shift_dist


def clip_pt_using_normal_form(pt, n_0, dist, min_dist):
    # compute offset
    offset_line = _offset_pt_to_normal_form_line(pt, n_0, dist)
    # check if correction necessary
    if np.abs(offset_line) > min_dist:
        # compute correction
        if offset_line >= 0:
            correction = offset_line - min_dist
        else:
            correction = offset_line + min_dist
        # apply correction
        pt = _shift_pt_to_normal_form_line(pt, n_0, -correction)
    return pt


def clip_bbox_using_line(bbox, line_pts_arr, min_dist=128/2.):
    # get normal form of line
    n_0, dist = normal_form_from_pts(line_pts_arr[0], line_pts_arr[1])
    # compute distance to line and decide if pt needs to be shifted
    pt_list = []
    # iterate over two bounding box coordinates
    for pt in [bbox[:2], bbox[2:]]:
        pt = clip_pt_using_normal_form(pt, n_0, dist, min_dist)
        pt_list.append(pt)

    return np.concatenate(pt_list)


def clip_bbox_using_line_segmentation(bbox, line_pts_arr, skeleton, min_dist=128/2.):
    # get normal form of line
    n_0, dist = normal_form_from_pts(line_pts_arr[0], line_pts_arr[1])

    # use bbox boundaries to crop pts from line segmentation
    # seg_line_pts = np.nonzero(skeleton[:, int(bbox[0]):int(bbox[2])])[0]
    # faster but more exclusive (probably worth the speedup)
    seg_line_pts = np.nonzero(skeleton[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])])[0] + int(bbox[1])

    dist_delta = 0
    if len(seg_line_pts) > 3:
        # compute average y location of segmentation line pts
        seg_line_cy = np.mean(seg_line_pts)
        # determine local distance delta from linear model to skeleton
        pt = [(bbox[0] + bbox[2]) / 2., seg_line_cy]
        dist_delta = _offset_pt_to_normal_form_line(pt, n_0, dist)
        # correct normal form of line [n_0, dist] using delta, ie. alter dist
        dist = dist + dist_delta
    #print dist_delta, dist - dist_delta, dist

    # compute distance to line and decide if pt needs to be shifted
    pt_list = []
    # iterate over two bounding box coordinates
    for pt in [bbox[:2], bbox[2:]]:
        pt = clip_pt_using_normal_form(pt, n_0, dist, min_dist)
        pt_list.append(pt)

    return np.concatenate(pt_list)


def dist_pt_line(pt, lpt1, lpt2):
    # compute squared 'perpendicular distance'
    # pt is point
    # lpt are line points
    # returns minimum (perpendicular) distance from point to line

    # assumes line representation of form y = a * x + b
    (a, b) = line_params_from_pts(lpt1, lpt2)
    # from: energy based geometric model fitting (2010)
    # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Another_formula
    return (np.abs(pt[1]-a*pt[0]-b)/np.sqrt(a**2+1))**2


def dist_lineseg_line(spt1, spt2, lpt1, lpt2):
    # computes the distance between line (unbounded) and line segment (bounded)
    # spt are line segment points
    # lpt are line points
    # returns minimum distance from line segment to line
    return min(dist_pt_line(spt1, lpt1, lpt2),
               dist_pt_line(spt2, lpt1, lpt2))


def assign_line_segments_to_lines(line_segs, line_hypos, x1=10):
    # get line pts from polar lines
    polar_lines = line_hypos.groupby('label').mean()[['angle', 'dist']].values
    line_pts = line_pts_from_polar_line(polar_lines[:, 0], polar_lines[:, 1], x1=x1)
    line_pts = np.transpose(np.concatenate(line_pts))
    # get line segments
    line_seg_pts = np.stack(line_segs).reshape(len(line_segs), -1)

    # compute distance between line segments and lines
    X2_dist = cdist(line_pts, line_seg_pts,
                    lambda lpts, spts: dist_lineseg_line(spts[:2], spts[2:], lpts[:2], lpts[2:]))
    # assign line segments to nearest line
    ls_labels = np.argmin(X2_dist, axis=0)

    return ls_labels


# associate line segments with segments


def associate_segments_with_lines(lbl_ind, line_segs, ls_labels, group2line):
    # create markers from line segments
    im_marker = np.zeros_like(lbl_ind)
    for line, li in zip(line_segs, ls_labels):
        p0, p1 = line
        rr, cc = draw.line(p0[1], p0[0], p1[1], p1[0])
        im_marker[rr, cc] = int(group2line[li]) + 1  # avoid background class
    # plt.imshow(im_marker)

    # use water shed to assign labels to segments
    distance = ndi.distance_transform_edt(lbl_ind)
    segm_labels = watershed(-distance, im_marker, mask=lbl_ind)
    return segm_labels, im_marker


# map segment lbls to image resolution (deal with network architecture with offset)

def compute_image_label_map(segm_labels, image_shape, padding=0):
    # collect patch boxes and their labels
    list_patch_boxes, list_patch_labels = [], []
    for lbl_idx in np.unique(segm_labels):
        if lbl_idx > 0:
            # for index compute coordinate boxes
            vx, vy = np.where(segm_labels == lbl_idx)
            patch_boxes = label_map2image(vy, vx, segm_labels.shape[::-1]).astype(int)
            # append
            list_patch_boxes.append(patch_boxes)
            list_patch_labels.append(patch_boxes.shape[0] * [lbl_idx])
            # vis_detections(center_im, patch_boxes, max_vis=200, labels="")

    patch_boxes = np.concatenate(list_patch_boxes, axis=0)
    patch_labels = np.concatenate(list_patch_labels, axis=0)
    # vis_detections(center_im, np.concatenate(list_patch_boxes, axis=0) , max_vis=1000, labels="")

    # create segmentation map from boxes and labels
    seg_canvas = np.zeros(image_shape[:2])
    for bb, lbl in zip(patch_boxes, patch_labels):
        pad = padding
        bb[:2] = bb[:2] - pad
        bb[2:] = bb[2:] + pad
        # print patch_box, patch_lbl
        seg_canvas[bb[1]:bb[3], bb[0]:bb[2]] = lbl

    return seg_canvas


def compute_line_length_from_tl(group, stats, b=128.):  # 128 / (2 * 32)
    # collect widths
    widths = np.zeros(len(group))
    for ii, (sidx, sign_rec) in enumerate(group.iterrows()):
        widths[ii] = stats.get_sign_width(sign_rec.lbl, sign_width=1) * b
    # compute offsets and line length
    sign_xpos = widths.cumsum() - (widths / 2.)
    line_len = widths.sum()
    # add columns to group
    group['prior_line_len'] = np.rint(line_len)
    group.loc[group.index, 'prior_sign_xoff'] = np.rint(sign_xpos)
    group.loc[group.index, 'prior_sign_width'] = np.rint(widths)

    return group


##### full pipeline


def post_process_line_detections(lbl_ind_x, num_lines, len_min, len_max, verbose=True):
    # identify lines and merge them if too close together
    # line hypothesis are stored in line_hypos dataframe
    # line_hypos.label indicates which lines are grouped together(merged) -> line_hypo_agg

    # (0) perform skeletonization
    skeleton = skeletonize(lbl_ind_x)
    # skeleton = skeletonize_3d(lbl_ind)
    # skeleton = thin(lbl_ind)

    # (1) compute hough transform
    h, theta, d, theta_range, theta_range2 = compute_hough_transform(skeleton, skeleton)  # skeleton, lbl_ind_x,

    # (I) find peaks in hough transform
    num_peaks_factor = 1.9  # 1.5 1.6  v007: 1.9 v047: 2.5 # line detector dependent (VIP)
    hl_peak_threshold = (h.max() / float(len_max)) / 2. * len_min  # 2. # has impact on lenght of lines found
    accums, angles, dists = hough_line_peaks(h, theta, d, min_distance=1, min_angle=14,
                                             num_peaks=int(num_lines * num_peaks_factor),
                                             threshold=hl_peak_threshold)

    # ugly patch for hough_line_peaks shortcomings
    # in rare cases len(accums) != len(angles) or len(dists
    if len(accums) != len(angles):
        angles = accums
        dists = accums

    # (II) check if lines intersect close to the center and group them accordingly
    interval = [lbl_ind_x.shape[1] * 1 / 8., lbl_ind_x.shape[1] * 7 / 8.]
    X_dist = pdist(np.stack([angles, dists], axis=1), lambda l1, l2: do_intersect_in_interval(l1, l2, interval))
    labels = compute_group_labels_from_dists(X_dist).astype(int)
    if verbose:
        print('detected groups: {} | num lines: {}.'.format(len(np.unique(labels)), num_lines))
    # collect lines in dataframe
    line_hypos = pd.DataFrame({'accum': accums, 'angle': angles, 'dist': dists, 'label': labels})
    line_hypos_agg = line_hypos.groupby('label').mean()
    # add group diff column
    diffs = line_hypos_agg.dist.sort_values().diff()
    # compute interline median
    dist_interline_median = diffs.median()

    # (III) check if remaining groups are very close
    X_dist = pdist(np.stack([line_hypos_agg.angle.values, line_hypos_agg.dist.values], axis=1),
                   lambda l1, l2: nearby_and_near_parallel_2(l1, l2, dist_interline_median, interval))
    updated_labels = compute_group_labels_from_dists(X_dist).astype(int)
    # update dataframe and grouping
    line_hypos.label.replace(to_replace=line_hypos_agg.index.values, value=updated_labels, inplace=True)

    # (IV) re-group with updated labels and get line meta for later usage
    line_hypos_agg = line_hypos.groupby('label').mean()
    # group lines and remember index (needs to be here)
    group2line = line_hypos_agg.index.values
    # add column group dist diff
    diffs = line_hypos_agg.dist.sort_values().diff()
    diffs.name = 'group_diff'  # set name before join
    line_hypos = line_hypos.join(diffs, on='label')
    # add column group angle diff
    angle_diff = line_hypos_agg.sort_values('dist').angle.apply(np.rad2deg).diff()
    angle_diff.name = 'group_angle_diff'  # set name before join
    line_hypos = line_hypos.join(angle_diff, on='label')
    # compute interline median
    dist_interline_median = diffs.median()
    if verbose:
        print('Update: detected groups: {} | num lines: {}.'.format(len(line_hypos_agg), num_lines))

    # (V) label line detection segments according to line hypos
    # compute probabilistic hough transform for lines
    # line_segs = probabilistic_hough_line(skeleton, threshold=6, line_length=15,
    #                             line_gap=6, theta=basic_theta)
    line_length = 8  # v007: 8 v047: 5  # line detector dependent (VIP)
    if len_max < line_length:
        line_length = len_max
    line_segs = probabilistic_hough_line(skeleton, threshold=6, line_length=line_length,
                                         line_gap=6, theta=theta_range)
    if len(line_segs) > 0:
        # assign line segments to nearest line
        ls_labels = assign_line_segments_to_lines(line_segs, line_hypos)

        # associate segments with lines
        segm_labels, im_marker = associate_segments_with_lines(lbl_ind_x, line_segs, ls_labels, group2line)
    else:
        segm_labels, ls_labels = lbl_ind_x, None  # set segm_labels so that line_frag gets a shape

    return line_hypos, line_segs, segm_labels, ls_labels, dist_interline_median, group2line, h, theta, d, skeleton



