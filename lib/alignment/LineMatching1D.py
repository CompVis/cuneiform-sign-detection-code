from scipy.spatial.distance import cdist, seuclidean, euclidean, squareform
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from timeit import default_timer as timer

import opengm

from ..utils.bbox_utils import bb_intersection_over_union


class LineMatching1D(object):

    def __init__(self, tl_line_rec, region_det, line_rec, line_pts, stats, scale=1.0, sign_hypos=None, param_dict=None):
        # create graphical model from fragement
        self.stats = stats
        self.scale = scale
        self.scaled_sign_height = stats.tblSignHeight * scale
        self.min_sign_dist = self.scaled_sign_height / 2.  # distance between sign centers

        self.tl_line_rec = tl_line_rec

        # null hypothesis for signs in tl
        self.sign_hypos = sign_hypos

        # detections contained in rectangluar area around respective alignments
        # [ID, cx, cy, score, x1, y1, x2, y2, idx]
        self.region_det = region_det

        # init
        self.num_vars = len(self.tl_line_rec)
        self.num_relevant = 0
        self.max_cost = 1e10  # 1e11  # "inifinite" cost

        # only continue, if there is a sign in line to match
        if self.num_vars > 0:

            # compute num_lbls_per_var from detections
            ulbls, counts = np.unique(self.region_det[:, 0], return_counts=True)
            hypo_det_counts = np.array([counts[ulbls == item] if item in ulbls else 0 for item in self.tl_line_rec.lbl],
                                       dtype=int).squeeze()

            self.tl_line_rec['det_count'] = hypo_det_counts
            # optional: remove vars without detections
            if False:
                # self.tl_line_rec = self.tl_line_rec[hypo_det_counts > 0]
                self.tl_line_rec = self.tl_line_rec.iloc[np.where(hypo_det_counts > 0)]  # deal with scalar case of boolean indexing
                # hypo_det_counts = hypo_det_counts[hypo_det_counts > 0]

            # only continue, at least a single matching detection
            self.num_relevant = np.sum(counts[np.isin(ulbls, self.tl_line_rec.lbl)])
            if self.num_relevant > 0:
                # update num_vars
                self.num_vars = len(self.tl_line_rec)
                # opengm setup
                self.num_lbls_per_var = max(counts[np.isin(ulbls, self.tl_line_rec.lbl)]) + 1  # + 1 outlier detection
                var_space = np.ones(self.num_vars) * self.num_lbls_per_var
                self.gm = opengm.gm(var_space)

                # parameter setup
                if param_dict is not None:
                    self.params = param_dict
                else:
                    self.params = dict()

                    # extra settings
                    self.params['outlier_cost'] = 10
                    self.params['angle_long_range'] = True

                    # unary potentials
                    self.params['lambda_score'] = 0.3
                    self.params['sigma_score'] = 0.4

                    self.params['lambda_offset'] = 1  # currently offset used linearly without exp function
                    self.params['sigma_offset'] = 1   # lambda & sigma have no influence!

                    # pairwise binary potentials
                    self.params['lambda_p'] = 3  # 1
                    self.params['sigma_p'] = 3

                    self.params['lambda_angle'] = 2
                    self.params['sigma_angle'] = 0.6

                    self.params['lambda_iou'] = 2
                    self.params['sigma_iou'] = 0.4

                    # OPTIONAL: strong penalties for long range connections
                    if True:
                        self.params['lr_lambda_angle'] = 0.05
                        self.params['lr_sigma_angle'] = 0.1

                        self.params['lr_lambda_iou'] = 0.1
                        self.params['lr_sigma_iou'] = 0.05
                    else:
                        self.params['lr_lambda_angle'] = self.params['lambda_angle']
                        self.params['lr_sigma_angle'] = self.params['sigma_angle']

                        self.params['lr_lambda_iou'] = self.params['lambda_iou']
                        self.params['lr_sigma_iou'] = self.params['sigma_iou']

                # angle of hypothesis line
                self.b = line_pts[-1, :] - line_pts[0, :]
                # print 'hypo angle:', np.arctan2(self.b[1], self.b[0]) * (180 / np.pi), self.b

                # offset
                self.Xb = line_pts[0, :].reshape(1, -1)

                # define variance between line distance and sign distance - for seculidean and mahalanobis
                self.variance_p = np.array([1, 0.2], dtype=np.float)  # [8, 1] [1, 1]

                if False:
                    print('#syms:', len(self.tl_line_rec), 'max#dets_per_sym:', self.num_lbls_per_var - 1,
                          'relevant#dets:', self.num_relevant, 'total#dets:', self.region_det.shape[0])

                # print(np.vstack([self.fm_hypo_df.lbl, hypo_det_counts])).astype(int)
                # print self.fm_hypo_df

                # assemble potentials
                self.add_unary()
                self.add_pairwise()

    def add_unary(self):

        # for monitoring costs
        self.unary_score_ct = {}
        self.unary_offset_ct = {}
        self.unary_det_ct = {}
        # compute for later usage by alignment vector
        self.det_same_label_ct_idx = []

        # assemble unary potentials
        for vidx, (tl_sign_idx, fm_sign) in enumerate(self.tl_line_rec.iterrows()):
            lbl = int(fm_sign.lbl)
            # ctr = [float(fm_sign.ctr_l), float(fm_sign.ctr_r)]

            # print vidx, lbl, ctr
            # get boxes of certain lbl
            det_same_label = self.region_det[self.region_det[:, 0] == lbl]
            self.det_same_label_ct_idx.append(det_same_label[:, -1])
            # detection locations
            Xa = det_same_label[:, [1, 2]]

            # incorporate score
            unary_vec = np.ones(self.num_lbls_per_var) * self.max_cost

            U1, U2, U3 = [], [], []
            if det_same_label.shape[0] > 0:
                # compute partial cost (vectorized)
                U1 = 1 - det_same_label[:, 3]
                # since goal of matching is to incorporate low confidence detections,
                # linear contribution of score might be enough / otherwise penalize only low confidence below 0.01
                if True:
                    U1 = self.params['lambda_score'] * (np.exp(U1 / self.params['sigma_score']) - 1)

                # incorporate distance from hypothesis line
                # cx, cy
                U2 = np.zeros(len(U1))
                if False:  # disabled in favour of null hypo offset
                    if self.params['lambda_offset'] != 0:
                        U2 = cdist(Xa, self.Xb, lambda u, v: np.linalg.norm(np.cross(u - v, self.b))
                                                             / np.linalg.norm(self.b)) / self.min_sign_dist
                        # compute partial cost
                        U2 = self.params['lambda_offset'] * (np.exp(U2.squeeze() / self.params['sigma_offset']) - 1)

                # incorporate null hypothesis of signs
                U3 = np.zeros(len(U1))
                if self.params['lambda_offset'] != 0 and self.sign_hypos is not None:
                    # sign hypo location
                    X0 = self.sign_hypos[vidx, 0:2].reshape(1, -1)
                    # get sign width and set variance
                    if lbl in self.stats.sign_df.index:
                        sign_width = self.stats.get_sign_width(lbl)
                    else:
                        sign_width = 1
                    var = np.array([sign_width * 1, 1], dtype=np.float)
                    # compute pairwise distance
                    U3 = cdist(X0, Xa, metric='seuclidean', V=var) / self.min_sign_dist
                    # U3 = self.params['lambda_offset'] * (np.exp(U3.squeeze() / self.params['sigma_offset']) - 1)
                    # U3 = np.clip(U3, 0, 1e-5 * self.max_cost)

                # sum up cost and insert into unary vector (only replace values if there a detections
                unary_vec[:len(U1)] = U1 + U2 + U3

            # for outlier detection set specific unary cost
            unary_vec[-1] = self.params['outlier_cost']

            # add function and factor
            func_id = self.gm.addFunction(unary_vec)
            self.gm.addFactor(func_id, vidx)

            # for debugging
            # self.unary_score_ct.append(U1)
            # self.unary_offset_ct.append(U3)
            # self.unary_det_ct.append(det_same_label)
            self.unary_score_ct[vidx] = U1
            self.unary_offset_ct[vidx] = U3
            self.unary_det_ct[vidx] = det_same_label

    def add_pairwise(self):
        # assemble pairwise potentials
        # Assumption: vars are in order of symbols in line
        # ATTENTION: ORDER of fm_hypo_lbls is important for pairwise potential generation!!!

        self.pairwise_dist_ct = {}
        self.pairwise_angle_ct = {}
        self.pairwise_iou_ct = {}
        self.pairwise_long_range = {}
        for vidx in range(self.num_vars - 1):
            # setup basic matrix with maximum cost
            dist_mat = np.ones([self.num_lbls_per_var] * 2) * self.max_cost

            sym_lt = self.tl_line_rec.lbl.iat[vidx]
            sym_rt = self.tl_line_rec.lbl.iat[vidx + 1]
            # get boxes according to labels
            # [ID, cx, cy, score, x1, y1, x2, y2]
            det_sym_lt = self.region_det[self.region_det[:, 0] == sym_lt]
            det_sym_rt = self.region_det[self.region_det[:, 0] == sym_rt]

            # x2, cy
            # sym_lt_right_border = det_sym_lt[:,[6,2]]
            # x1, cy
            # sym_rt_left_border = det_sym_rt[:,[4,2]]

            # cx, cy
            sym_lt_right_border = det_sym_lt[:, [1, 2]]
            sym_rt_left_border = det_sym_rt[:, [1, 2]]

            # bboxes
            sym_lt_bboxes = det_sym_lt[:, 4:]
            sym_rt_bboxes = det_sym_rt[:, 4:]

            # compute pairwise distances between detections of lt and rt sym

            # 1) basic computation
            # X = cdist(sym_lt_right_border, sym_rt_left_border, metric='euclidean')
            X = cdist(sym_lt_right_border, sym_rt_left_border, metric='seuclidean', V=self.variance_p)
            # because vertical offset always depends on underlying rotation, mahalanobis should be used here
            # X = cdist(sym_lt_right_border, sym_rt_left_border, metric='mahalanobis', VI=self.VI)
            # reduce distances to normal scale and normalize with 10 * times sign_height
            X = ((X/self.scaled_sign_height) - 1)

            inX = X.copy()
            # compute partial cost
            X = self.params['lambda_p'] * (np.exp(X / self.params['sigma_p']) - 1)

            # 2) penalty for wrong side
            # if on wrong side, increase cost by factor 4  [is deprecated due to angle computation!!!]
            #X2 = cdist(sym_lt_right_border, sym_rt_left_border, lambda u, v: u[0] > v[0])
            #X[X2.astype(bool)] *= 5

            # 3) penalize distance only in x-dimension
            # X8 = cdist(sym_lt_right_border, sym_rt_left_border, lambda u, v: v[0] - u[0])
            # X8 = self.params['lambda_p'] * np.exp((self.min_sign_dist - X8) / self.params['sigma_p'])

            # incorporate angle
            # angle with x-axis: np.arctan((u[1]-v[1])/(u[0]-v[0]))
            # b=np.array([1,0])
            # angle between vectors less stable: acos(dot(v1, v2) / (norm(v1) * norm(v2)))
            # X3 = cdist(sym_lt_right_border, sym_rt_left_border,
            #           lambda u,v: np.arccos(np.dot(v-u,b) / (np.linalg.norm(v-u) * np.linalg.norm(b))))/pi
            # angle between vectors more numerical stable: atan2(norm(cross(a,b)), dot(a,b))
            X3 = cdist(sym_lt_right_border, sym_rt_left_border,
                       lambda u, v: np.arctan2(np.linalg.norm(np.cross(v - u, self.b)), np.dot(v - u, self.b))) / np.pi
            inX3 = X3.copy()
            # compute partial cost
            X3 = self.params['lambda_angle'] * (np.exp(X3 / self.params['sigma_angle']) - 1)

            # incorporate IoU
            X4 = cdist(sym_lt_bboxes, sym_rt_bboxes,
                       lambda u, v: bb_intersection_over_union(u, v))
            inX4 = X4.copy()
            # compute partial cost
            X4 = self.params['lambda_iou'] * (np.exp(X4 / self.params['sigma_iou']) - 1)

            # sum up cost and insert into dist_mat
            dist_mat[:X.shape[0], :X.shape[1]] = X + X3 + X4

            # for outlier class set pairwise cost to 0
            dist_mat[-1, :] = 0
            dist_mat[:, -1] = 0

            # avoid identity solutions
            if sym_lt == sym_rt:
                np.fill_diagonal(dist_mat, self.max_cost)

            # add function and factor
            func_id = self.gm.addFunction(dist_mat)
            self.gm.addFactor(func_id, [vidx, vidx + 1])

            # for debugging
            # self.pairwise_dist_ct[(vidx, vidx + 1)] = inX
            # self.pairwise_angle_ct[(vidx, vidx + 1)] = inX3
            # self.pairwise_iou_ct[(vidx, vidx + 1)] = inX4
            self.pairwise_dist_ct[(vidx, vidx + 1)] = X
            self.pairwise_angle_ct[(vidx, vidx + 1)] = X3
            self.pairwise_iou_ct[(vidx, vidx + 1)] = X4

            # in the case of angles add pairwise potentials for all possible combinations
            if self.params['angle_long_range']:
                # add combinations on the right of var
                # not necessary to add combinations on the left of var due to symmetry
                for vidx_rt in range(vidx + 2, self.num_vars):
                    sym_rt = self.tl_line_rec.lbl.iat[vidx_rt]
                    # detections
                    det_sym_rt = self.region_det[self.region_det[:, 0] == sym_rt]
                    # cx, cy
                    sym_rt_left_border = det_sym_rt[:, [1, 2]]
                    # bboxes
                    sym_rt_bboxes = det_sym_rt[:, 4:]
                    # incorporate angle
                    # angle between vectors more numerical stable: atan2(norm(cross(a,b)), dot(a,b))
                    XY3 = cdist(sym_lt_right_border, sym_rt_left_border,
                               lambda u, v: np.arctan2(np.linalg.norm(np.cross(v - u, self.b)),
                                                       np.dot(v - u, self.b))) / np.pi
                    # compute partial cost
                    XY3 = self.params['lr_lambda_angle'] * (np.exp(XY3 / self.params['lr_sigma_angle']) - 1)

                    # incorporate iou
                    XY4 = cdist(sym_lt_bboxes, sym_rt_bboxes,
                               lambda u, v: bb_intersection_over_union(u, v))

                    # compute partial cost
                    XY4 = self.params['lr_lambda_iou'] * (np.exp(XY4 / self.params['lr_sigma_iou']) - 1)

                    # sum up cost and insert into dist_mat
                    dist_mat[:XY3.shape[0], :XY3.shape[1]] = XY3 + XY4

                    # for outlier class set pairwise cost to 0
                    dist_mat[-1, :] = 0
                    dist_mat[:, -1] = 0

                    # avoid identity solutions
                    if sym_lt == sym_rt:
                        np.fill_diagonal(dist_mat, self.max_cost)

                    # add function and factor
                    func_id = self.gm.addFunction(dist_mat)
                    self.gm.addFactor(func_id, [vidx, vidx_rt])

                    # for debugging
                    self.pairwise_long_range[(vidx, vidx_rt)] = XY3 + XY4  # XY3, XY4, XY3 + XY4

    def run_inference(self):
        # only continue, if there is a sign/detection in line to match
        if len(self.tl_line_rec) > 0 and self.num_relevant > 0:

            if False:
                # basic belief propagation (slower)
                bfprop = opengm.inference.BeliefPropagation(gm=self.gm)
            if True:
                # TRWS: https://github.com/opengm/opengm/blob/master/src/interfaces/python/opengm/inference/pyTrws.cxx
                # default params: https://github.com/opengm/opengm/blob/master/src/interfaces/python/opengm/inference/param/trws_external_param.hxx
                parameter = opengm.InfParam(steps=200)
                bfprop = opengm.inference.TrwsExternal(gm=self.gm, accumulator='minimizer', parameter=parameter)

            #start = timer()
            bfprop.infer()
            #run_time = timer() - start
            #print('{}'.format(run_time))

            # get and save labeling
            self.labeling = bfprop.arg()
            self.tl_line_rec['lbl_arg'] = bfprop.arg()

            # get raw energy and check if inference failed
            self.raw_energy = self.gm.evaluate(bfprop.arg())
            self.inference_failed = self.raw_energy > self.num_vars * self.params['outlier_cost']

            # get energy, normalize by num_vars * outlier_cost
            # worst case should be outliers only
            max_line_cost = self.num_vars * self.params['outlier_cost']
            # clip energy, because inference sometimes fails !?
            self.energy = min(self.raw_energy, max_line_cost) / float(max_line_cost)
            # attributes cost to individual assignments (selected detections) and normalize using outlier_cost
            self.tl_line_rec['nE'] = np.around(self.compute_labeling_energy() / self.params['outlier_cost'], decimals=2)

            if self.inference_failed:
                # all outlier
                self.tl_line_rec['aligned_det_idx'] = -1
                self.tl_line_rec['region_det_idx'] = -1
            else:
                # compute actual alignments with respect to original detections indices
                self._compute_global_alignments()
                # compute alignments with respect to region detection indices
                self._compute_region_alignments()

    def _compute_global_alignments(self):
        alignments = np.zeros((len(self.tl_line_rec), 1), dtype=int)
        for i, lbl in enumerate(self.labeling):
            if lbl != (self.num_lbls_per_var - 1) and len(self.det_same_label_ct_idx[i]) > 0:
                alignments[i] = self.det_same_label_ct_idx[i][lbl]
            else:
                # outlier
                alignments[i] = -1
        # set values in dataframe
        self.tl_line_rec['aligned_det_idx'] = alignments.astype(int)

    def _compute_region_alignments(self):
        alignments = np.zeros((len(self.tl_line_rec), 1), dtype=int)
        for ii, global_det_idx in enumerate(self.tl_line_rec.aligned_det_idx.values):
            if global_det_idx != -1:
                # map global to region detection index
                alignments[ii] = np.where(self.region_det[:, -1] == global_det_idx)[0]
            else:
                # outlier detection
                alignments[ii] = -1
        # set values in dataframe
        self.tl_line_rec['region_det_idx'] = alignments.astype(int)

    def get_region_alignments(self):
        # maybe I should also return the self.tl_line_rec.index
        # problem arises if self.tl_line_rec is changed inside LineMatching1D
        if 'region_det_idx' in self.tl_line_rec.columns:
            return self.tl_line_rec.region_det_idx.values
        else:
            return []

    def visualize_matching(self, input_im, sign_hypos, ax=None):
        # only continue, if there is a sign in line to match
        if len(self.tl_line_rec) > 0:

            # select detections using alignment index
            alignments = self.get_region_alignments()
            aligned = self.region_det[alignments[alignments >= 0], 1:3]

            if ax is None:
                fig, ax = plt.subplots(figsize=(12, 8))
            # plot hypo
            ax.plot(sign_hypos[:, 0], sign_hypos[:, 1], '*b', markersize=10, label='null hypo')
            ax.plot(aligned[:, 0], aligned[:, 1], 'oy', markersize=8, label='gm aligned detections')
            # plot tablet
            ax.imshow(input_im, cmap=plt.cm.Greys_r)

            # annotate
            for i, pos_idx in enumerate(self.tl_line_rec.iloc[alignments >= 0].pos_idx.values):
                ax.annotate(pos_idx, (aligned[i, 0], aligned[i, 1]), fontsize=15)

            ax.legend(shadow=True, fancybox=True)
            ax.axis('off')
            # plt.show()

    # energy marginal computation

    def _get_unary_cost(self, unary_dict, vidx, didx):
        unary = unary_dict[vidx]
        if len(unary) > 0:
            return unary.flatten()[didx]
        else:
            # in cases there inference fails and labeling is out of bounds
            return self.max_cost

    def _get_pairwise_val(self, pairwise_dict, idx0, idx1):
        outlier_lbl = self.num_lbls_per_var - 1
        pairwise = pairwise_dict[idx0, idx1]
        didx0 = self.labeling[idx0]
        didx1 = self.labeling[idx1]
        if didx0 != outlier_lbl and didx1 != outlier_lbl:
            if pairwise.size > 0:
                return pairwise[didx0, didx1]
            else:
                # in cases there inference fails and labeling is out of bounds
                return self.max_cost
        else:
            return 0

    def _get_pairwise_cost(self, pairwise_dict, vidx):
        # deal with boundary cases
        if vidx == self.num_vars - 1:
            return self._get_pairwise_val(pairwise_dict, vidx - 1, vidx)
        elif vidx == 0:
            return self._get_pairwise_val(pairwise_dict, vidx, vidx + 1)
        else:
            return (self._get_pairwise_val(pairwise_dict, vidx - 1, vidx)
                    + self._get_pairwise_val(pairwise_dict, vidx, vidx + 1))

    def _get_lr_pairwise_cost(self, lr_pairwise_dict, vidx):
        energy = 0
        for vidx_rt in range(vidx + 2, self.num_vars):
            energy += self._get_pairwise_val(lr_pairwise_dict, vidx, vidx_rt)
        return energy

    def compute_unary_cost(self):
        list_unary = [self.unary_score_ct, self.unary_offset_ct]
        outlier_lbl = self.num_lbls_per_var - 1
        u_marginals = np.zeros_like(self.labeling, dtype=np.float)
        for vidx, didx in enumerate(self.labeling):
            if didx != outlier_lbl:
                for unary_dict in list_unary:
                    u_marginals[vidx] += self._get_unary_cost(unary_dict, vidx, didx)
            else:
                u_marginals[vidx] += self.params['outlier_cost']

        return u_marginals

    def compute_pairwise_cost(self):
        list_pairwise = [self.pairwise_angle_ct, self.pairwise_dist_ct, self.pairwise_iou_ct]

        p_marginals = np.zeros_like(self.labeling, dtype=np.float)
        if len(self.labeling) > 1:  # only compute if there are any pairs
            for vidx, dvidx in enumerate(self.labeling):
                for pairwise_dict in list_pairwise:
                    p_marginals[vidx] += self._get_pairwise_cost(pairwise_dict, vidx)
        return p_marginals

    def compute_pairwise_cost_lr(self):
        lr_pairwise_dict = self.pairwise_long_range

        p_marginals = np.zeros_like(self.labeling, dtype=np.float)
        if len(self.labeling) > 1:  # only compute if there are any pairs
            for vidx, dvidx in enumerate(self.labeling):
                p_marginals[vidx] += self._get_lr_pairwise_cost(lr_pairwise_dict, vidx)
        return p_marginals

    def compute_labeling_energy(self):
        # compute an energy vector that attributes cost to individual labels
        # if the output vector summed up, this equals the un-normalized energy

        # deal with case when inference failed
        if self.inference_failed:

            return self.max_cost
        else:
            u_marginals = self.compute_unary_cost()
            p_marginals = self.compute_pairwise_cost()
            plr_marginals = self.compute_pairwise_cost_lr()

            return u_marginals + (p_marginals/2. + plr_marginals)

