# config

# version of initial hypos
sign_model_version=v129

# suffix for reference
suffix=""   # _v2


# test
#echo "test on test and train set"
#python script_gen_initial_hypo.py -c test train --sign_model_version=$sign_model_version --suffix=$suffix

# generate initial placed detections
echo "test saa01 saa05 saa06 saa08 saa10 saa13 saa16 train"
python script_gen_initial_hypo.py -c test saa01 saa05 saa06 saa08 saa10 saa13 saa16 train --sign_model_version=$sign_model_version --suffix=$suffix



# rename files
# (as generated annotations are concatenated this avoids duplicates if the script is rerun)
initial_sign_model_version="$sign_model_version""_initial_hypos"

echo "rename " $initial_sign_model_version "gen files"
python script_rename_files.py --sign_model_version=$initial_sign_model_version


