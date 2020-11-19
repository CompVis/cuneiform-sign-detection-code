import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt

from operator import itemgetter

from scipy.stats import norm
from scipy import ndimage as ndi
from scipy.spatial.distance import pdist, cdist, squareform

from LineFragment import LineFragment


# LINES - TRANSLITERATION ALIGNMENT PROBLEM
# associate lines with transliteration lines


# OPTION 0) use line_models sorted by dist as basic alignment

def align_lines_tl_by_sort(line_hypos, tl_df):
    # use tl_line_indices
    tl_line_indices = tl_df.line_idx.unique()

    # extend or cut if too short or long respectively
    diff_len = line_hypos.label.nunique() - len(tl_line_indices)
    if diff_len > 0:
        last_idx = tl_line_indices[-1] + 1
        tl_line_indices = np.concatenate([tl_line_indices, range(last_idx,  last_idx + diff_len)])
    else:
        tl_line_indices = tl_line_indices[:line_hypos.label.nunique()]

    # print tl_line_indices, line_hypos.groupby('label').mean().sort_values('dist').index

    # find basic alignment by sorting (enumerate line models sorted according to dist)
    tl_line_assignment = pd.DataFrame({'tl_line': tl_line_indices,  #  np.arange(line_hypos.label.nunique())
                                       'hypo_line_lbl': line_hypos.groupby('label').mean().sort_values('dist').index})

    # add tl_line column in line_hypos using join on line_hypos
    return line_hypos.join(tl_line_assignment.set_index('hypo_line_lbl'), on='label')


# OPTION 1) use ground truth annotations as alignment
# use gt line annotations (implicit update tl_line column in line_hypos)
# (unreliable, because gt line annotations and transliteration are not necessarily aligned themselves!)

def align_lines_tl_by_ground_truth(line_hypos, tl_df):
    # update tl_line with gt_line_idx (set nan to -1)
    #line_hypos['tl_line'] = line_hypos['gt_line_idx'].fillna(-1)
    line_hypos = line_hypos.assign(tl_line=line_hypos['gt_line_idx'].fillna(-1))
    # if there are more gt_lines than tl_lines ...
    # gt_line_idx that are not in tl are replaced with -1
    not_tl_line_idx = ~line_hypos['tl_line'].isin(tl_df.line_idx.unique())
    line_hypos.loc[not_tl_line_idx, 'tl_line'] = -1
    return line_hypos


# OPTION 2) adopted from GALE-CHURCH algorithm for sentence alignment
# relies on line lengths only

norm_logsf = norm.logsf
LOG2 = math.log(2)

AVERAGE_CHARACTERS = 1
VARIANCE_CHARACTERS = 6.8

BEAD_COSTS = {(1, 1): 0, (2, 1): 1000,  # (2, 1): 230
              (1, 2): 1000, (0, 1): 230,
              (1, 0): 230, (2, 2): 2000}  # (1, 0): 450

# BEAD_COSTS = {(1, 1): 0, (2, 1): 230, (1, 2): 230, (0, 1): 450,
#                (1, 0): 450, (2, 2): 440}


def length_cost(sx, sy, mean_xy, variance_xy):
    """
    Code from https://github.com/alvations/gachalign:
    Calculate length cost given 2 sentence. Lower cost = higher prob.

    The original Gale-Church (1993:pp. 81) paper considers l2/l1 = 1 hence:
    delta = (l2-l1*c)/math.sqrt(l1*s2)

    If l2/l1 != 1 then the following should be considered:
    delta = (l2-l1*c)/math.sqrt((l1+l2*c)/2 * s2)
    substituting c = 1 and c = l2/l1, gives the original cost function.
    """
    lx, ly = sum(sx), sum(sy)
    m = (lx + ly * mean_xy) / 2

    try:
        delta = (lx - ly * mean_xy) / math.sqrt(m * variance_xy)
    except ZeroDivisionError:
        return float('-inf')

    return - 100 * (LOG2 + norm_logsf(abs(delta)))


def _align(x, y, mean_xy, variance_xy, bead_costs):
    """
    The minimization function to choose the sentence pair with
    cheapest alignment cost.
    """
    m = {}
    for i in range(len(x) + 1):
        for j in range(len(y) + 1):
            if i == j == 0:
                m[0, 0] = (0, 0, 0)
            else:
                m[i, j] = min((m[i - di, j - dj][0] + length_cost(x[i - di:i], y[j - dj:j], mean_xy, variance_xy)
                               + bead_cost, di, dj)
                              for (di, dj), bead_cost in BEAD_COSTS.iteritems()
                              if i - di >= 0 and j - dj >= 0)

    i, j = len(x), len(y)
    while True:
        (c, di, dj) = m[i, j]
        if di == dj == 0:
            break
        yield (i - di, i), (j - dj, j)
        i -= di
        j -= dj


def align_lines_tl_by_gale_church(tl_df, line_hypos, variance_characters=3.0):
    # updates line_hypos with tl_line idx
    # actually uses line_hypos_agg

    # get line lengths
    tl_line_len = tl_df.groupby('line_idx').mean().prior_line_len
    det_line_len = line_hypos.groupby('label').mean().sort_values('dist').accum
    # define input

    cx = tl_line_len.values
    cy = det_line_len.values

    # use detection line lengths to normalize (better range than tl lengths)
    max_char = int(cy.max())

    # normalize
    cx /= cx.max()
    cx *= max_char
    #cy /= cy.max()
    #cy *= max_char
    bc = BEAD_COSTS

    # iterate over aligned pairs
    for (i1, i2), (j1, j2) in reversed(list(_align(cx, cy, 1.0, variance_characters, bc))):
        # print (i1, i2), (j1, j2)
        # print (tl_line_len.index[i1:i2].values, det_line_len.index[j1:j2].values)
        # check if line_hypo exists
        if len(det_line_len.index[j1:j2].values) > 0:
            tl_line_idx = -1
            if len(tl_line_len.index[i1:i2].values) > 0:
                tl_line_idx = int(tl_line_len.index[i1:i2].values[0])
            # assign tl line idx to detected line
            line_hypos.loc[line_hypos.label.isin(det_line_len.index[j1:j2].values), 'tl_line'] = tl_line_idx
    # return cx, cy
    return line_hypos


# OPTION 3) adopted from Bleualign algorithm for sentence alignment
# relies on matching score between tl null hypothesis and sign detections (sign detector)
# the problem is to align hypo_line_indices (detected lines) with tl_line_indices (transliteration lines)
# all information required is contained in line fragment

# a) make sure that score_mat forms valid positive weights for edges in graph
# b) get matching score matrix with shape=[len(hypo_line_indices), len(tl_line_indices)]
# c) alignment consists of segments that are connected diagonally


def compute_bleu_score_mat(hypo_line_indices, tl_line_indices, line_frag):
    # ransac score
    score_mat = cdist(hypo_line_indices.reshape(-1, 1), tl_line_indices.reshape(-1, 1),
                   lambda a_idx, b_idx: line_frag.compute_bleu_score(a_idx.squeeze(), b_idx.squeeze()))  # 5/5, 4/1
    # score in range [0, 1], but order needs to be reversed
    score_mat = 1 - score_mat
    return score_mat


def compute_ransac_score_mat(hypo_line_indices, tl_line_indices, line_frag):
    # ransac score
    score_mat = cdist(hypo_line_indices.reshape(-1, 1), tl_line_indices.reshape(-1, 1),
                   lambda a_idx, b_idx: line_frag.compute_ransac_score(a_idx.squeeze(), b_idx.squeeze(),
                                                                       max_dist_thresh=2, dist_weight=1))  # 5/5, 4/1
    # score in range [0, 1], but order needs to be reversed
    score_mat = 1 - score_mat
    return score_mat


def compute_matching_score_mat(hypo_line_indices, tl_line_indices, line_frag):
    # line matching score
    score_mat = cdist(hypo_line_indices.reshape(-1, 1), tl_line_indices.reshape(-1, 1),
                   lambda a_idx, b_idx: line_frag.compute_line_matching_score(a_idx.squeeze(), b_idx.squeeze()))

    # score in range [0, 1], but order needs to be reversed
    score_mat = 1 - score_mat
    return score_mat


# use this if you want to implement your own similarity score
def eval_sents_dummy(translist, targetlist, max_alternatives=3):
    scoredict = {}

    for testID, testSent in enumerate(translist):
        scores = []

        for refID, refSent in enumerate(targetlist):
            score = 100 - abs(len(testSent) - len(refSent))  # replace this with your own similarity score
            if score > 0:
                scores.append((score, refID, score))
        # sorted by first item in tuple (i.e. score)
        scoredict[testID] = sorted(scores, key=itemgetter(0), reverse=True)[:max_alternatives]

    return scoredict


# follow the backpointers in score matrix to extract best path of 1-to-1 alignments
def extract_best_path(pointers):

    i = len(pointers)-1
    j = len(pointers[0])-1
    pointer = ''
    best_path = []

    while i >= 0 and j >= 0:
        pointer = pointers[i][j]
        if pointer == '^':
            i -= 1
        elif pointer == '<':
            j -= 1
        elif pointer == 'match':
            best_path.append((i, j))
            i -= 1
            j -= 1

    best_path.reverse()
    return best_path


# dynamic programming search for best path of alignments (maximal score)
def pathfinder(translist, targetlist, scoremat):  # scoredict

    # add an extra row/column to the matrix and start filling it from 1,1 (to avoid exceptions for first row/column)
    matrix = [[0 for column in range(len(targetlist)+1)] for row in range(len(translist)+1)]
    pointers = [['' for column in range(len(targetlist))] for row in range(len(translist))]

    for i in range(len(translist)):
        for j in range(len(targetlist)):

            best_score = matrix[i][j+1]
            best_pointer = '^'

            score = matrix[i+1][j]
            if score > best_score:
                best_score = score
                best_pointer = '<'

            #if np.abs(j - i) < 5:  # distance from diagonal
            score = scoremat[i, j] + matrix[i][j]
            if score > best_score:
                best_score = score
                best_pointer = 'match'

            matrix[i+1][j+1] = best_score
            pointers[i][j] = best_pointer

    bleualign = extract_best_path(pointers)
    return bleualign


def align_lines_tl_by_score(line_hypos, line_frag, visualize=True):
    # alignment based on longest path through score mat (topological sort)
    assert 'tl_line' in line_hypos.columns, "tl_line needs to be set (e.g. use align by sort"

    # get assignment space (cartesian product of tl_line_indices and hypo_line_indices)
    hypo_line_indices, tl_line_indices = line_frag.get_alignment_space()
    # print(hypo_line_indices, tl_line_indices)

    align_opts = [1, 0, 0, 0, 0, 0]  # align + ransac, most accurate, slow  [NORMAL]
    #align_opts = [0, 0, 1, 0, 0, 0]  # bleu + ransac, a little less accurate, fast  [use with high number of detections]
    #align_opts = [0, 0, 0, 0, 0, 1]  # bleu
    #align_opts = [0, 0, 0, 0, 1, 0]  # ransac
    #align_opts = [0, 0, 0, 1, 0, 0]  # align

    assert(np.sum(align_opts) <= 1)

    # prepare score mats
    if align_opts[0]:
        score_mats = [compute_ransac_score_mat(hypo_line_indices, tl_line_indices, line_frag),
                      compute_matching_score_mat(hypo_line_indices, tl_line_indices, line_frag)]
        title_strs = ['ransac', 'gm matching']
        multi_score = True
    if align_opts[1]:
        score_mats = [compute_bleu_score_mat(hypo_line_indices, tl_line_indices, line_frag),
                      compute_matching_score_mat(hypo_line_indices, tl_line_indices, line_frag)]
        title_strs = ['bleu', 'gm matching']  # bleu
        multi_score = True
    if align_opts[2]:
        score_mats = [compute_ransac_score_mat(hypo_line_indices, tl_line_indices, line_frag),
                      compute_bleu_score_mat(hypo_line_indices, tl_line_indices, line_frag)]
        title_strs = ['ransac', 'bleu']
        multi_score = True
    if align_opts[3]:
        score_mats = [compute_matching_score_mat(hypo_line_indices, tl_line_indices, line_frag)]
        title_strs = ['gm matching']
        multi_score = False
    if align_opts[4]:
        score_mats = [compute_ransac_score_mat(hypo_line_indices, tl_line_indices, line_frag)]
        title_strs = ['ransac']
        multi_score = False
    if align_opts[5]:
        score_mats = [compute_bleu_score_mat(hypo_line_indices, tl_line_indices, line_frag)]
        title_strs = ['bleu']
        multi_score = False


    if visualize:
        # prepare plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 15, 5
        ax = axes.ravel()

    best_paths = []
    for i, score_mat in enumerate(score_mats):
        best_path = pathfinder(hypo_line_indices, tl_line_indices, score_mat)
        best_paths.append(best_path)
        path_pts = np.asarray(best_path)

        if visualize:
            # plot score mats with shortest path
            if len(path_pts) > 0:
                ax[i].plot(path_pts[:, 1], path_pts[:, 0])
                ax[i].plot(path_pts[:, 1], path_pts[:, 0], 'cd')
            ax[i].imshow(score_mat)
            ax[i].set_title(title_strs[i])

    # compute joint
    if multi_score:
        path_pts = np.asarray(sorted(set(best_paths[0]).intersection(best_paths[1])))
        # path_pts = np.asarray(best_paths[1])  # use gm_matching only
    else:
        path_pts = np.asarray(best_paths[0])

    if visualize:
        # plot score mats with shortest path
        if len(path_pts) > 0:
            ax[2].plot(path_pts[:, 1], path_pts[:, 0])
            ax[2].plot(path_pts[:, 1], path_pts[:, 0], 'cd')
        ax[2].imshow((score_mats[0] + score_mats[1]) / 2.)
        ax[2].set_title('joint')

    if len(path_pts) > 0:
        # map path through score mat back to line_idx (because score mat idx not necessarily equal score mat idx)
        # hl_indices = hypo_line_indices[path_pts[:, 0]]  # already equals index to dataframe
        tl_indices = tl_line_indices[path_pts[:, 1]]

        # print tl_line_assignment, path_pts[:, 0], tl_indices

        # create assignment table for join
        basic_index = line_frag.line_hypos.tl_line.sort_values().unique()
        tl_line_assignment = pd.DataFrame({'hypo_tl_line': basic_index, 'tl_line_update': -np.ones_like(basic_index)})
        tl_line_assignment.loc[path_pts[:, 0], 'tl_line_update'] = tl_indices
        # join line_hypos on tl_line
        line_hypos['tl_line'] = line_hypos.join(tl_line_assignment.set_index('hypo_tl_line'), on='tl_line')[
            'tl_line_update']

    return line_hypos, path_pts


####  full pipeline to solve the line-transliteration alignment problem ####

def compute_line_tl_alignment(line_hypos, tl_df, gt_line_assignment, segm_labels, stats, center_im, sign_detections,
                              visualize=True, align_opt=[False, False, True]):

    path_pts = None

    # BASIC:
    # use line_models sorted by dist as basic alignment
    line_hypos = align_lines_tl_by_sort(line_hypos, tl_df)

    # OPTION I:
    # find basic alignment using line lengths
    # apply Gale-Church algorithm (implicit update tl_line column in line_hypos)
    if align_opt[0]:  # False
        line_hypos = align_lines_tl_by_gale_church(tl_df, line_hypos, variance_characters=6.0)

    # OPTION II:
    # use gt line annotations (implicit update tl_line column in line_hypos)
    if align_opt[1]:  # False
        if len(gt_line_assignment) > 0:
            line_hypos = align_lines_tl_by_ground_truth(line_hypos, tl_df)

    # OPTION III:
    # alignment based on longest path through score mat (topological sort)
    if align_opt[2]:  # True
        # create line fragment (tl_line should be assigned before!)
        line_frag = LineFragment(line_hypos, segm_labels, tl_df, stats, center_im, sign_detections)
        # compute lines tl alignment based on score
        (line_hypos, path_pts) = align_lines_tl_by_score(line_hypos, line_frag, visualize=visualize)

    return line_hypos, path_pts



## GT function


def gt_align_lines_tl_by_ed(line_gt, visualize=True):
    # alignment based on longest path through score mat (topological sort)

    # get assignment space (cartesian product of tl_line_indices and gt_line_indices)
    gt_line_indices, tl_line_indices = line_gt.get_alignment_space()

    # prepare score mats
    score_mats = [compute_bleu_score_mat(gt_line_indices, tl_line_indices, line_gt)]
    title_strs = ['edit distance']
    multi_score = False

    if visualize:
        # prepare plot
        fig, axes = plt.subplots(1, 1, figsize=(15, 5), squeeze=False)  # 1,3
        ax = axes.ravel()

    best_paths = []
    for i, score_mat in enumerate(score_mats):
        best_path = pathfinder(gt_line_indices, tl_line_indices, score_mat)
        best_paths.append(best_path)
        path_pts = np.asarray(best_path)

        if visualize:
            # plot score mats with shortest path
            if len(path_pts) > 0:
                ax[i].plot(path_pts[:, 1], path_pts[:, 0])
                ax[i].plot(path_pts[:, 1], path_pts[:, 0], 'cd')
            ax[i].imshow(score_mat)
            ax[i].set_title(title_strs[i])

    # compute joint
    if multi_score:
        path_pts = np.asarray(sorted(set(best_paths[0]).intersection(best_paths[1])))
        # path_pts = np.asarray(best_paths[1])  # use gm_matching only
    else:
        path_pts = np.asarray(best_paths[0])

    if len(path_pts) > 0:
        # map path through score mat back to line_idx (because score mat idx not necessarily equal score mat idx)
        # gt_indices = gt_line_indices[path_pts[:, 0]]  # already equals index to dataframe
        tl_indices = tl_line_indices[path_pts[:, 1]]
        # print tl_line_assignment, path_pts[:, 0], tl_indices

        lines_df = line_gt.lines_df
        # create assignment table for join
        #basic_index = lines_df.tl_line.sort_values().unique()  # this is not necessary, because gt_line_idx
        basic_index = lines_df.gt_line_idx.sort_values().unique()
        tl_line_assignment = pd.DataFrame({'gt_tl_line': basic_index, 'tl_line_update': -np.ones_like(basic_index)})
        tl_line_assignment.loc[path_pts[:, 0], 'tl_line_update'] = tl_indices
        # print tl_line_assignment, path_pts

        # join line_hypos on tl_line
        line_gt.lines_df['tl_line'] = lines_df.join(tl_line_assignment.set_index('gt_tl_line'), on='gt_line_idx')['tl_line_update']

    return line_gt, path_pts


