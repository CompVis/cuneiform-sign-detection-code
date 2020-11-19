import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi


def show_line_skeleton(lbl_ind_x, skeleton):
    # display results
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), sharex=True, sharey=True,
                             subplot_kw={'adjustable': 'box-forced'})
    ax = axes.ravel()

    ax[0].imshow(lbl_ind_x, cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title('original', fontsize=20)

    ax[1].imshow(skeleton, cmap=plt.cm.gray)
    ax[1].axis('off')
    ax[1].set_title('skeleton', fontsize=20)

    ax[2].imshow(ndi.label(skeleton, structure=np.ones((3, 3)))[0], cmap=plt.cm.spectral)
    ax[2].axis('off')
    ax[2].set_title('skeleton', fontsize=20)

    fig.tight_layout()


def show_hough_transform_w_lines(lbl_ind_x, center_im, h, theta, d, line_hypos, color):
    # Generating figure 1
    fig, axes = plt.subplots(1, 3, figsize=(15, 6),
                             subplot_kw={'adjustable': 'box-forced'})  # (25, 15)
    ax = axes.ravel()

    ax[0].imshow(center_im, cmap='gray')
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    ax[1].imshow(np.log(1 + h),
                 extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
                 cmap='gray', aspect=1 / 1.5)
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')

    ax[2].imshow(lbl_ind_x, cmap='gray')

    for idx, line_rec in line_hypos.groupby('label').mean().iterrows():
        angle = line_rec.angle
        dist = line_rec.dist
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - lbl_ind_x.shape[1] * np.cos(angle)) / np.sin(angle)
        ax[2].plot((0, lbl_ind_x.shape[1]), (y0, y1), '-', color=color[int(idx)], linewidth=2)

    ax[2].set_xlim((0, lbl_ind_x.shape[1]))
    ax[2].set_ylim((lbl_ind_x.shape[0], 0))
    ax[2].set_axis_off()
    ax[2].set_title('Detected lines')

    # ax[2].imshow(lbl_ind, cmap='gray')
    # ax[2].set_title('Input image')
    # ax[2].set_axis_off()


def show_probabilistic_hough(lbl_ind_x, center_im, line_segs, ls_labels, group2line, color):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax = axes.ravel()

    ax[0].imshow(center_im, cmap='gray')
    ax[0].set_title('Input image')

    # ax[1].imshow(lbl_ind_x, cmap='gray')
    # ax[1].set_title('line det')

    ax[1].imshow(lbl_ind_x * 0)
    for line, li in zip(line_segs, ls_labels):
        p0, p1 = line
        ax[1].plot((p0[0], p1[0]), (p0[1], p1[1]), color=color[int(group2line[li])], linewidth=2)
        ax[1].text(p0[0], p0[1], '{}'.format(group2line[li]),
                   bbox=dict(facecolor='blue', alpha=0.5), fontsize=8, color='white')
    ax[1].set_xlim((0, lbl_ind_x.shape[1]))
    ax[1].set_ylim((lbl_ind_x.shape[0], 0))
    ax[1].set_title('Probabilistic Hough')


def show_line_segms(image_label_overlay, segm_labels):
    fig, axes = plt.subplots(1, 2, figsize=(15, 9))  # 25, 15
    ax = axes.ravel()

    ax[0].imshow(image_label_overlay, cmap='gray')
    ax[0].set_title('Input image')

    ax[1].imshow(segm_labels)
    ax[1].set_title('Line segments')







