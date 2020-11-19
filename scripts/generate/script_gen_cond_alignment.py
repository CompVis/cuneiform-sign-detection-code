## IMPORTANT: When training with hypo data, make sure to use same threshold as while generating it
# otherwise detections can overlap with conditional hypos -> duplicate entries !!

import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
import time

relative_path = '../../'
# ensure that parent path is on the python path in order to have all packages available
import sys, os

parent_path = os.path.join(os.getcwd(), relative_path)
parent_path = os.path.realpath(parent_path)  # os.path.abspath(...)
sys.path.insert(0, parent_path)


from lib.datasets.segments_dataset import CuneiformSegments

from lib.transliteration.sign_labels import get_label_list

from lib.evaluations.sign_evaluation_gt import BBoxAnnotations
from lib.evaluations.line_evaluation import LineAnnotations, visualize_line_segments_with_labels

from lib.utils.path_utils import clean_cdli, prepare_data_gen_folder_slim, make_folder

from lib.alignment.run_gen_cond_alignments import gen_cond_hypo_alignments


# argument passing

parser = argparse.ArgumentParser(description='Generate aligned detections.')
parser.add_argument('-c', '--collections', nargs='+', default=['saa01'])
parser.add_argument('--sign_model_version')
parser.add_argument('--suffix', default='', help='e.g. "_v2"')
parser.add_argument('--min_dets_inline', default=4, type=int, help='min number of detections inline')
parser.add_argument('--max_dist_det', default=3, type=int, help='positional offset from alignments')

# parse
args = parser.parse_args()
# show
print(args.collections, args.sign_model_version, args.suffix)

# assign
gen_model_version = args.sign_model_version
sign_model_version = gen_model_version
collections = args.collections
suffix = args.suffix

min_dets_inline = args.min_dets_inline
max_dist_det = args.max_dist_det

# config
generate_and_save = True  # generate and save conditional hypos

concat_and_save = True  # append anno_df with conditional hypos and save (For testing set to False)

line_model_version = 'v007'

num_classes = 240

visualize_hypos = False


# cond config (only relevant if ncompl computed)
ncompl_thresh = 10  # disable: -1 (no need to disable, if there is no ncompl computed)
smooth_y = True  # True smooth y coordinate according to alignments

print('ncompl: {} | min_dets: {} | max_dist: {}'.format(ncompl_thresh, min_dets_inline, max_dist_det))

# ### Load Datasets

# load lbl list
lbl_list = get_label_list(relative_path + 'data/newLabels.json')


# ### Run gen conditional hypos


for saa_version in collections:
    print('collection: <><>{}<><>'.format(saa_version))

    ### Load generated annotations
    annotation_file = '{}results/results_ssd/{}/line_generated_bboxes_refined80_{}.csv'.format(relative_path,
                                                                                               sign_model_version,
                                                                                               saa_version)
    column_names = ['imageName', 'folder', 'image_path', 'label', 'newLabel', 'x1', 'y1', 'x2', 'y2', 'width', 'height',
                    'seg_idx',
                    'line_idx', 'pos_idx', 'det_score', 'm_score', 'align_ratio', 'nms_keep', 'compl', 'ncompl']
    anno_df = pd.read_csv(annotation_file, engine='python', names=column_names)
    anno_df['bbox'] = np.vstack([np.rint(anno_df.x1.values), np.rint(anno_df.y1.values),
                                 np.rint(anno_df.x2.values), np.rint(anno_df.y2.values)]).transpose().astype(
        int).tolist()
    # only use classes in range
    anno_df = anno_df[anno_df.newLabel < num_classes]

    # completeness: fill nan values in a way that avoids filtering
    # (this should make code work, if no completeness computed)
    anno_df.compl = anno_df.compl.fillna(50)
    anno_df.ncompl = anno_df.ncompl.fillna(100)

    # print stats
    print(len(anno_df))
    print(anno_df.newLabel.value_counts().describe())

    ### Get collection dataset and annotations
    # load segments dataset  # preload_segments=True
    dataset = CuneiformSegments(transform=None, target_transform=None, relative_path=relative_path,
                                collection=saa_version)

    # load annotations for collection
    bbox_anno = BBoxAnnotations(dataset.collection, relative_path=dataset.relative_path)
    lines_anno = LineAnnotations(dataset.collection, coll_scales=dataset.assigned_segments_df.scale,
                                 relative_path=relative_path)

    ### Create folder
    # create path to file that stores generated training data
    res_path_base = '{}results/results_ssd/{}_cond_hypos{}'.format(relative_path, gen_model_version, suffix)
    train_data_ext_file, collection_subfolder = prepare_data_gen_folder_slim(saa_version, res_path_base)

    # filter collection dataset - OPTIONAL
    didx_list = range(len(dataset))
    # didx_list = didx_list[:10]  # 10:11 11:-2   # 5:6  #46

    ### Create conditional hypo alignments
    gen_cond_hypo_alignments(didx_list, dataset, bbox_anno, lines_anno, anno_df, relative_path, saa_version,
                             collection_subfolder, train_data_ext_file, lbl_list, generate_and_save,
                             min_dets_inline, ncompl_thresh, smooth_y, max_dist_det,
                             line_model_version, visualize_hypos)

    ### Store
    if concat_and_save:
        # load, concatenate and save back to csv
        # annotation_file = '{}pytorch/results_ssd/{}_cond_hypos/line_generated_bboxes_{}.csv'.format(relative_path, sign_model_version, saa_version)
        anno_hypos_df = pd.read_csv(train_data_ext_file, engine='python', names=column_names)
        new_anno_df = anno_df.drop(columns=['bbox']).append(anno_hypos_df)
        new_anno_df.to_csv(train_data_ext_file, header=False, index=False)
        print(new_anno_df.newLabel.value_counts().describe())


