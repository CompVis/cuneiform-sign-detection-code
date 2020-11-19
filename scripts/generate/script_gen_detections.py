import argparse

import pandas as pd
import torch


relative_path = '../../'
# ensure that parent path is on the python path in order to have all packages available
import sys, os

parent_path = os.path.join(os.getcwd(), relative_path)
parent_path = os.path.realpath(parent_path)  # os.path.abspath(...)
sys.path.insert(0, parent_path)


from lib.datasets.cunei_dataset_segments import CuneiformSegments, get_segment_meta
from lib.models.trained_model_loader import get_fpn_ssd_net
from lib.detection.run_gen_ssd_detection import gen_ssd_detections


# argument passing

parser = argparse.ArgumentParser(description='Generate aligned detections.')
parser.add_argument('-c', '--collections', nargs='+', default=['saa01'])
parser.add_argument('--sign_model_version')
parser.add_argument('--test_min_score_thresh', type=float, default=0.01)

# parse
args = parser.parse_args()
# show
print(args.collections, args.sign_model_version)

# assign
model_version = args.sign_model_version
collections = args.collections
test_min_score_thresh = args.test_min_score_thresh

# collections = ['test', 'train', 'saa01',
#                'saa05',
#                'saa08',
#                'saa10', 'saa13', 'saa16']  #
#
# collections += ['train']

only_annotated = False

# store detections for re-use
save_detections = True

# show detections
show_detections = False


arch_type = 'mobile'  # resnet, mobile
arch_opt = 1
width_mult = 0.625  # 0.5 0.625

crop_shape = [600, 600]
tile_shape = [600, 600]

num_classes = 240

with_64 = False
create_bg_class = False  # for v018: True


#test_min_score_thresh = 0.015  # 0.01 0.05 0.4
test_nms_thresh = 0.5  # 0.5 0.6

eval_ovthresh = 0.5  # 0.4

# ### Load Model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

fpnssd_net = get_fpn_ssd_net(model_version, device, arch_type, with_64, arch_opt, width_mult,
                             relative_path, num_classes, num_c=1)

print(fpnssd_net)


### Test net
loc_preds, cls_preds = fpnssd_net(torch.randn(1, 1, 1024, 1024).to(device))
print(loc_preds.size(), cls_preds.size())

# ### Predict with SSD detector


for saa_version in collections:
    print('collection: <><>{}<><>'.format(saa_version))

    ### Get collection dataset and annotations
    dataset = CuneiformSegments(collections=[saa_version], relative_path=relative_path,
                                only_annotated=only_annotated, preload_segments=False)

    # filter collection dataset - OPTIONAL
    didx_list = range(len(dataset))
    # didx_list = didx_list[5:9]

    ### Generate ssd detections
    (list_seg_ap,
     list_seg_name_with_anno) = gen_ssd_detections(didx_list, dataset, saa_version, relative_path,
                                                   model_version, fpnssd_net, with_64, create_bg_class, device,
                                                   test_min_score_thresh, test_nms_thresh, eval_ovthresh,
                                                   save_detections, show_detections)

    ### Evaluate on annotations (if available)
    if False:
        in_data = {'tablet_name': list_seg_name_with_anno, 'mAP': list_seg_ap}
        res_df = pd.DataFrame(in_data)
        print(res_df)

        print(res_df.mAP.mean())




