# config

# version of aligned detections
sign_model_version=v191ft01_hp04

# suffix for reference
suffix=""   # _v2

# min number of detections inline
min_dets_inline=1

# pos offset from alignments
max_dist_det=10


# test
#echo "test on test and train set"
#python script_gen_cond_alignment.py -c test train --sign_model_version=$sign_model_version

# https://unix.stackexchange.com/questions/103920/parallelize-a-bash-for-loop
# run cond alignment in parallel
echo "test saa01 saa05"
python script_gen_cond_alignment.py -c test saa01 saa05 --sign_model_version=$sign_model_version --min_dets_inline=$min_dets_inline --max_dist_det=$max_dist_det --suffix=$suffix &

echo "saa08"
python script_gen_cond_alignment.py -c saa08 --sign_model_version=$sign_model_version --min_dets_inline=$min_dets_inline --max_dist_det=$max_dist_det --suffix=$suffix &

echo "saa10"
python script_gen_cond_alignment.py -c saa10 --sign_model_version=$sign_model_version --min_dets_inline=$min_dets_inline --max_dist_det=$max_dist_det --suffix=$suffix &

echo "saa13 saa16 train"
python script_gen_cond_alignment.py -c saa13 saa16 train --sign_model_version=$sign_model_version --min_dets_inline=$min_dets_inline --max_dist_det=$max_dist_det --suffix=$suffix &


# wait for all scripts to finish
wait


