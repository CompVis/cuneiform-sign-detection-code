import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, cdist, squareform


def show_lines_tl_alignment(lbl_ind_x, center_im, line_hypos, color):
    # Generating figure 1
    fig, axes = plt.subplots(1, 2, figsize=(15, 6),   # (25, 10)
                             subplot_kw={'adjustable': 'box-forced'})
    ax = axes.ravel()

    ax[0].imshow(center_im, cmap='gray')
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    ax[1].imshow(lbl_ind_x, cmap='gray')

    for idx, line_rec in line_hypos.groupby('label').mean().iterrows():
        angle = line_rec.angle
        dist = line_rec.dist
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - lbl_ind_x.shape[1] * np.cos(angle)) / np.sin(angle)
        ax[1].plot((0, lbl_ind_x.shape[1]), (y0, y1), '-', color=color[int(idx)], linewidth=2)
        ax[1].text(0, y0, '{}'.format(int(line_rec.tl_line)),
                   bbox=dict(facecolor='blue', alpha=0.5), fontsize=8, color='white')

    ax[1].set_xlim((0, lbl_ind_x.shape[1]))
    ax[1].set_ylim((lbl_ind_x.shape[0], 0))
    ax[1].set_axis_off()
    ax[1].set_title('Detected lines / Assigned tl line idx')


def show_score_mats_with_paths(assigned_tl_indices, hypo_line_indices, tl_line_indices, line_frag):
    # Generating figure 1
    fig, axes = plt.subplots(1, 3, figsize=(15, 6),
                             subplot_kw={'adjustable': 'box-forced'})
    ax = axes.ravel()

    # weak score
    X_dist = cdist(assigned_tl_indices.reshape(-1, 1), assigned_tl_indices.reshape(-1, 1),
                   lambda a_idx, b_idx: line_frag.compute_weak_score(a_idx.squeeze(), b_idx.squeeze()))
    ax[0].imshow(X_dist, cmap='gray')
    ax[0].set_title('weak score')
    print(np.diag(X_dist))

    # ransac score
    # X_dist = cdist(assigned_tl_indices.reshape(-1, 1), assigned_tl_indices.reshape(-1, 1),
    X_dist = cdist(hypo_line_indices.reshape(-1, 1), tl_line_indices.reshape(-1, 1),
                   lambda a_idx, b_idx: line_frag.compute_ransac_score(a_idx.squeeze(), b_idx.squeeze(),
                                                                       max_dist_thresh=2, dist_weight=1))  # 5/5, 4/1
    ax[1].imshow(X_dist, cmap='gray_r')
    ax[1].set_title('ransac score')
    print(np.diag(X_dist))

    # line matching score
    X_dist = cdist(hypo_line_indices.reshape(-1, 1), tl_line_indices.reshape(-1, 1),
                   lambda a_idx, b_idx: line_frag.compute_line_matching_score(a_idx.squeeze(), b_idx.squeeze()))
    ax[2].imshow(X_dist, cmap='gray_r')  # vmin=0, vmax=1
    ax[2].set_title('line matching score')
    print(np.diag(X_dist))


