import argparse
import glob
import shutil

relative_path = '../../'
# ensure that parent path is on the python path in order to have all packages available
import sys, os
parent_path = os.path.join(os.getcwd(), relative_path)
parent_path = os.path.realpath(parent_path)  # os.path.abspath(...)
sys.path.insert(0, parent_path)


# argument passing
parser = argparse.ArgumentParser(description='Generate aligned detections.')
parser.add_argument('--sign_model_version')
parser.add_argument('--coll_name', default='', help='saa, ransac, testEXT')

# parse
args = parser.parse_args()
# show
print(args.sign_model_version, args.coll_name)

# assign
sign_model_version = args.sign_model_version
coll_name = args.coll_name


# select relevant files
gen_folder = '' # 'gen_scoreth05nms50/'  # ''  'gen_scoreth01nms50/'
gen_folder = '{}results/results_ssd/{}/{}/'.format(relative_path, sign_model_version, gen_folder)
gen_files = [fn.split('/')[-1] for fn in glob.glob(gen_folder + 'line_generated_bboxes_{}*'.format(coll_name)) if 'refined80' not in fn]
print(gen_files)

# rename files
for gen_file in gen_files:
    shutil.move(gen_folder + gen_file, gen_folder + gen_file.replace('bboxes', 'bboxes_refined80'))

