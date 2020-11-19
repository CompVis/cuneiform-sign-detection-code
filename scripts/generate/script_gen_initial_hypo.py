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

from lib.alignment.run_gen_null_hypo_alignments import gen_null_hypo_alignments


# argument passing

parser = argparse.ArgumentParser(description='Generate initial placed detections.')
parser.add_argument('-c', '--collections', nargs='+', default=['saa01'])
parser.add_argument('--sign_model_version')
parser.add_argument('--suffix', default='', help='e.g. "_v2"')

# parse
args = parser.parse_args()
# show
print(args.collections, args.sign_model_version, args.suffix)

# assign
gen_model_version = args.sign_model_version
sign_model_version = gen_model_version
collections = args.collections
suffix = args.suffix


# config
generate_and_save = True  # generate and save initial hypos

line_model_version = 'v007'

num_classes = 240

visualize_hypos = False


# ### Load Datasets

# load lbl list
lbl_list = get_label_list(relative_path + 'data/newLabels.json')


# ### Run gen initial hypos

for saa_version in collections:
    print('collection: <><>{}<><>'.format(saa_version))

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
    res_path_base = '{}results/results_ssd/{}_initial_hypos{}'.format(relative_path, gen_model_version, suffix)
    train_data_ext_file, collection_subfolder = prepare_data_gen_folder_slim(saa_version, res_path_base)

    # filter collection dataset - OPTIONAL
    didx_list = range(len(dataset))
    # didx_list = didx_list[:10]  # 10:11 11:-2   # 5:6  #46

    ### Create initial hypo
    gen_null_hypo_alignments(didx_list, dataset, bbox_anno, lines_anno, relative_path, saa_version,
                             collection_subfolder, train_data_ext_file, lbl_list, generate_and_save,
                             line_model_version, visualize_hypos)

    ### Print stats
    if generate_and_save:
        column_names = ['imageName', 'folder', 'image_path', 'label', 'newLabel', 'x1', 'y1', 'x2', 'y2', 'width', 'height',
                        'seg_idx', 'line_idx', 'pos_idx', 'det_score', 'm_score', 'align_ratio', 'nms_keep']
        anno_hypos_df = pd.read_csv(train_data_ext_file, engine='python', names=column_names)
        print(anno_hypos_df.newLabel.value_counts().describe())

