# config

# version of sign detector
sign_model_version=v191ft01

# minimum confidence to keep detection
test_min_score_thresh=0.01  # 0.015 0.02


# test
#echo "test on test and train set"
#python script_gen_detections.py -c test train --sign_model_version=$sign_model_version --test_min_score_thresh=$test_min_score_thresh

# generate detections
echo "test saa01 saa05 saa06 saa08 saa10 saa13 saa16 train"
python script_gen_detections.py -c test saa01 saa05 saa06 saa08 saa10 saa13 saa16 train --sign_model_version=$sign_model_version --test_min_score_thresh=$test_min_score_thresh
