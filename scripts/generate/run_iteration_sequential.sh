###
# 0) config

# version of sign detector
sign_model_version=v191ft01

# hyper-parameter version
hp_version=hp04

# collections used for generation
# ds_version=(test train)   # test
ds_version=(test saa01 saa05 saa06 saa08 saa10 saa13 saa16 train)  # full run



echo $sign_model_version ${ds_version[@]}

###
# 1) generate detections

test_min_score_thresh=0.01  # 0.015 0.02

echo "gen detections: " $model_version ${ds_version[@]}
python script_gen_detections.py -c ${ds_version[@]} --sign_model_version=$sign_model_version --test_min_score_thresh=$test_min_score_thresh

###
# 2) align detections

python script_gen_alignment.py -c ${ds_version[@]} --sign_model_version=$sign_model_version --hyperparam_version=$hp_version

###
# 3) rename alignments

align_model_version="$sign_model_version""_""$hp_version"

echo "rename " $align_model_version "gen files"
python script_rename_files.py --sign_model_version=$align_model_version

###
# 4) cond align detections (placed and aligned)

suffix=""   # _v2
min_dets_inline=1  # 1 min number of detections inline
max_dist_det=10  # 10 pos offset from alignments

python script_gen_cond_alignment.py -c ${ds_version[@]} --sign_model_version=$align_model_version --min_dets_inline=$min_dets_inline --max_dist_det=$max_dist_det --suffix=$suffix

###
# 5) rename cond_alignments

cond_sign_model_version="$align_model_version""_cond_hypos"

echo "rename " $cond_sign_model_version "gen files"
python script_rename_files.py --sign_model_version=$cond_sign_model_version
