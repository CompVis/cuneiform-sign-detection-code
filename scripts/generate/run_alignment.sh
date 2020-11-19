# config

# version of sign detector
sign_model_version=v191ft01

# hyper-parameter version
hp_version=hp04  # hp04 hp05  hp00


# test
#echo "test on test and train set"
#python script_gen_alignment.py -c test train --sign_model_version=$sign_model_version --hyperparam_version=$hp_version

# https://unix.stackexchange.com/questions/103920/parallelize-a-bash-for-loop
# run alignment in parallel
echo "test saa01 saa05"
python script_gen_alignment.py -c test saa01 saa05 --sign_model_version=$sign_model_version --hyperparam_version=$hp_version &

echo "saa08"
python script_gen_alignment.py -c saa08 --sign_model_version=$sign_model_version --hyperparam_version=$hp_version &

echo "saa10"
python script_gen_alignment.py -c saa10 --sign_model_version=$sign_model_version --hyperparam_version=$hp_version &

echo "saa13 saa16 train"
python script_gen_alignment.py -c saa13 saa16 train --sign_model_version=$sign_model_version --hyperparam_version=$hp_version &
# wait for all scripts to finish
wait

# rename here
# or run cond alignmet




