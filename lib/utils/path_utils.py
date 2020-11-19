import os


# file names, folders, paths

def make_folder(res_path):
    # create folder, if it does not exist
    if not os.path.exists(res_path):
        os.makedirs(res_path)


def prepare_data_gen_folder(relative_path, sign_model_version, collection_name, res_folder_name='results'):
    # create path to file that stores generated training data
    res_path = '{}pytorch/{}/{}'.format(relative_path, res_folder_name, sign_model_version)
    train_data_ext_file = '{}/line_generated_bboxes_{}.csv'.format(res_path, collection_name)
    collection_subfolder = '{}/images/'.format(collection_name)
    # create folder, if necessary
    make_folder(res_path)
    # remove generated file, if it exists
    if os.path.isfile(train_data_ext_file):
        os.remove(train_data_ext_file)

    return train_data_ext_file, collection_subfolder, res_path


def prepare_data_gen_folder_slim(collection_name, res_path_base):
    # create path to file that stores generated training data

    train_data_ext_file = '{}/line_generated_bboxes_{}.csv'.format(res_path_base, collection_name)
    collection_subfolder = '{}/images/'.format(collection_name)
    # create folder, if necessary
    make_folder(res_path_base)
    # remove generated file, if it exists
    if os.path.isfile(train_data_ext_file):
        os.remove(train_data_ext_file)

    return train_data_ext_file, collection_subfolder


def clean_cdli(cdli_str):
    # remove Vs, Rs
    out_str = cdli_str.replace("Vs", "")
    out_str = out_str.replace("Rs", "")
    return out_str
