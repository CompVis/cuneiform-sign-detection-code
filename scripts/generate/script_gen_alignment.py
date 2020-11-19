import argparse

# torch
import torch
import torchvision

# addons
from tqdm import tqdm


relative_path = '../../'  # ''../../'
# ensure that parent path is on the python path in order to have all packages available
import sys, os

parent_path = os.path.join(os.getcwd(), relative_path)
parent_path = os.path.realpath(parent_path)  # os.path.abspath(...)
sys.path.insert(0, parent_path)


from lib.models.trained_model_loader import get_line_net_fcn

from lib.datasets.segments_dataset import CuneiformSegments

from lib.transliteration.sign_labels import get_label_list

from lib.evaluations.config import cfg, cfg_from_file, cfg_from_list
from lib.evaluations.line_evaluation import LineAnnotations
from lib.evaluations.sign_evaluation_gt import BBoxAnnotations

from lib.utils.transform_utils import UnNormalize
from lib.utils.path_utils import clean_cdli, prepare_data_gen_folder_slim, make_folder

from lib.alignment.run_gen_alignments import gen_alignments


# argument passing

parser = argparse.ArgumentParser(description='Generate aligned detections.')
parser.add_argument('-c', '--collections', nargs='+', default=['saa01'])
parser.add_argument('--sign_model_version')
parser.add_argument('--hyperparam_version', default='hp00')

# parse
args = parser.parse_args()
# show
print(args.collections, args.sign_model_version, args.hyperparam_version)

# assign
sign_model_version = args.sign_model_version
collections = args.collections


if args.hyperparam_version == 'hp04':
    hp_version = '_hp04'
    # ('lambda_score': 12.0)
    param_dict = {'angle_long_range': True,
                  'lambda_angle': 2,
                  'lambda_iou': 0.4,  # 2,
                  'lambda_offset': 1,
                  'lambda_p': 5,  # 3
                  'lambda_score': 12.0,  # 0.3,
                  'lr_lambda_angle': 0.2,  # 0.05,
                  'lr_lambda_iou': 1.5,  # 0.1
                  'lr_sigma_angle': 0.1,
                  'lr_sigma_iou': 0.05,
                  'outlier_cost': 25,  # 10,
                  'sigma_angle': 0.6,
                  'sigma_iou': 0.4,
                  'sigma_offset': 1,
                  'sigma_p': 3,
                  'sigma_score': 0.88,  # 0.4
                  'refined': True}
elif args.hyperparam_version == 'hp05':
    hp_version = '_hp05'
    # ('lambda_score': 11.5 instead of 14.3)
    param_dict = {'angle_long_range': True,
                  'lambda_angle': 2,
                  'lambda_iou': 0.4,  # 2,
                  'lambda_offset': 1,
                  'lambda_p': 5,  # 3
                  'lambda_score': 11.5,  # 0.3,
                  'lr_lambda_angle': 0.2,  # 0.05,
                  'lr_lambda_iou': 1.5,  # 0.1
                  'lr_sigma_angle': 0.1,
                  'lr_sigma_iou': 0.05,
                  'outlier_cost': 25,  # 10,
                  'sigma_angle': 0.6,
                  'sigma_iou': 0.4,
                  'sigma_offset': 1,
                  'sigma_p': 3,
                  'sigma_score': 0.88,  # 0.4
                  'refined': True}
else:
    hp_version = '_hp00'
    param_dict = {'angle_long_range': True,
                  'lambda_angle': 2,
                  'lambda_iou': 2,
                  'lambda_offset': 1,
                  'lambda_p': 3,  # 1
                  'lambda_score': 0.3,
                  'lr_lambda_angle': 0.05,
                  'lr_lambda_iou': 0.1,
                  'lr_sigma_angle': 0.1,
                  'lr_sigma_iou': 0.05,
                  'outlier_cost': 10,
                  'sigma_angle': 0.6,
                  'sigma_iou': 0.4,
                  'sigma_offset': 1,
                  'sigma_p': 3,
                  'sigma_score': 0.4,
                  'refined': True}

# config
generate_and_save = True

show_alignments = False

model_version = 'v007'

gpu_id = 0  # 0 1 all

pre_config = ['TEST.TP_MIN_OVERLAP', 0.5, 'TEST.NMS', 0.5]
# ensure config is loaded before dependent functions are used/ instantiated
cfg_from_list(pre_config)


# load lbl list
lbl_list = get_label_list(relative_path + 'data/newLabels.json')

# ### Config Data Augmentation

data_layer_params = dict(batch_size=[128, 16],
                         img_channels=1,
                         gray_mean=[0.5],  # 0.5 #0.488
                         gray_std=[1.0],  # 1.0 # 0.231
                         num_classes=2
                         )

num_classes = data_layer_params['num_classes']
num_c = data_layer_params['img_channels']
gray_mean = data_layer_params['gray_mean']
gray_std = data_layer_params['gray_std']


re_transform = torchvision.transforms.Compose([
    UnNormalize(mean=gray_mean, std=gray_std),
    torchvision.transforms.ToPILImage(),
])
re_transform_rgb = torchvision.transforms.Compose([
    UnNormalize(mean=gray_mean * 3, std=gray_std * 3),
    torchvision.transforms.ToPILImage(),
])

# ### Load Model

# use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if 0:
    model_fcn = get_line_net_fcn(model_version, device, num_classes=num_classes, num_c=num_c)
    print(model_fcn)
else:
    model_fcn = None


# ### Generate alignments
for saa_version in tqdm(collections):
    print('collection: <><>{}<><>'.format(saa_version))

    ### Create folder
    # create path to file that stores generated training data
    res_path_base = '{}results/results_ssd/{}{}'.format(relative_path, sign_model_version, hp_version)
    train_data_ext_file, collection_subfolder = prepare_data_gen_folder_slim(saa_version, res_path_base)

    ### Get collection dataset and annotations
    dataset = CuneiformSegments(transform=None, target_transform=None, relative_path=relative_path,
                                collection=saa_version)

    # load annotations for collection
    bbox_anno = BBoxAnnotations(dataset.collection, relative_path=dataset.relative_path)
    lines_anno = LineAnnotations(dataset.collection,
                                 coll_scales=dataset.assigned_segments_df.scale,
                                 relative_path=relative_path)

    # filter collection dataset - OPTIONAL
    didx_list = range(len(dataset))
    # didx_list = didx_list[5:6]  # 11:-2   # 5:6  #46  # :200

    ### Generate line hypothesis
    gen_alignments(didx_list, dataset, bbox_anno, lines_anno, relative_path, saa_version, re_transform,
                   sign_model_version, model_fcn, device,
                   generate_and_save, show_alignments, collection_subfolder, train_data_ext_file, lbl_list,
                   use_precomp_lines=True,
                   param_dict=param_dict)


