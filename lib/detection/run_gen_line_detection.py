import numpy as np
from tqdm import tqdm

from ..datasets.cunei_dataset_segments import CuneiformSegments, get_segment_meta
from ..detection.line_detection import (prepare_transliteration, preprocess_line_input, apply_detector)
from ..utils.path_utils import make_folder

from skimage.morphology import skeletonize


def gen_line_detections(didx_list, dataset, saa_version, relative_path,
                        line_model_version, model_fcn, re_transform, device,
                        save_line_detections):

    # for seg_im, seg_idx in dataset:
    # iterate over segments
    for didx in tqdm(didx_list, desc=saa_version):
        # print(didx)
        seg_im, gt_boxes, gt_labels = dataset[didx]

        # access meta
        seg_rec = dataset.get_seg_rec(didx)
        image_name, scale, seg_bbox, _, view_desc = get_segment_meta(seg_rec)
        res_name = "{}{}".format(image_name, view_desc)

        # make seg image is large enough for line detector
        if seg_im.size[0] > 224 and seg_im.size[1] > 224:

            # prepare input
            inputs = preprocess_line_input(seg_im, 1, shift=0)
            center_im = re_transform(inputs[4])  # to pil image
            center_im = np.asarray(center_im)  # to numpy

            try:
                # apply network
                output = apply_detector(inputs, model_fcn, device)
                # visualize_net_output(center_im, output, cunei_id=1, num_classes=2)
                # plt.show()

                # prepare output
                outprob = np.mean(output, axis=0)
                lbl_ind = np.argmax(outprob, axis=0)

                lbl_ind_x = lbl_ind.copy()
                lbl_ind_x[np.max(outprob, axis=0) < 0.7] = 0  # 7

                lbl_ind_80 = lbl_ind.copy()
                lbl_ind_80[np.max(outprob, axis=0) < 0.8] = 0    # remove squeeze() from outprob in order to fix a bug!

                # save line detections
                if save_line_detections:
                    # line result folder
                    line_res_path = "{}results/results_line/{}/{}".format(relative_path, line_model_version, saa_version)
                    make_folder(line_res_path)

                    # save lbl_ind_x
                    outfile = "{}/{}_lbl_ind.npy".format(line_res_path, res_name)
                    np.save(outfile, lbl_ind_x.astype(bool))

                    if False:
                        # compute skeleton
                        skeleton = skeletonize(lbl_ind_x)

                        # save skeleton
                        outfile = "{}/{}_skeleton.npy".format(line_res_path, res_name)
                        np.save(outfile, skeleton.astype(bool))
            except Exception as e:
                # Usually CUDA error: out of memory
                print res_name, e.message, e.args

