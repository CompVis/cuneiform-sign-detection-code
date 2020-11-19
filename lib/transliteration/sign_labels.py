import json
import numpy as np


def get_label_list(path_to_lbl_file='../../data/newLabels.json'):
    # get list that maps old -> new

    # load label list
    with open(path_to_lbl_file) as json_data:
        lbl_list = json.load(json_data)
    return lbl_list


def get_lbl2lbl(path_to_lbl_file):
    # get list that maps new -> old
    # actually using lbl_list with index function works as well !

    # load label list
    lbl_list = np.asarray(get_label_list(path_to_lbl_file))
    # print np.unique(lbl_list)
    # reverse (assume mapping is unique)
    lbl2lbl = np.zeros(len(np.unique(lbl_list)), )  # 240
    for (i, val) in enumerate(lbl_list):
        lbl2lbl[val] = i  # new -> old
    # since mapping is not unique for 0, need to set manually to background
    lbl2lbl[0] = 0
    return lbl2lbl

