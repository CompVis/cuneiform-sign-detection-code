import pandas as pd
from ..utils.path_utils import *


class TransliterationSet:

    def __init__(self, collections=[], relative_path='../../'):
        # load list of coll_tl_df
        list_coll_tl_df = []
        for collection in collections:
            coll_tl_file = '{}data/transliterations/transliterations_{}.csv'.format(relative_path, collection)
            # check if transliteration exists
            if os.path.isfile(coll_tl_file):
                print('Transliteration file {} found!'.format(coll_tl_file))
                # load transliteration
                coll_tl_df = pd.read_csv(coll_tl_file)
                # select subset of columns
                coll_tl_df = coll_tl_df[['segm_idx', 'tablet_CDLI', 'train_label', 'mzl_label', 'line_idx', 'pos_idx', 'status']]
                coll_tl_df['lbl'] = coll_tl_df['train_label']
                coll_tl_df['mzl_lbl'] = coll_tl_df['mzl_label']
            else:
                print('Transliteration file {} NOT found!'.format(coll_tl_file))
                coll_tl_df = pd.DataFrame()
            # append coll_tl_df to list
            list_coll_tl_df.append(coll_tl_df)
        # make accessible
        self.collections = collections
        self.list_coll_tl_df = list_coll_tl_df

    def get_tl_df(self, seg_rec, verbose=True):
        # init empty tl
        num_lines = 0
        tl_df = pd.DataFrame()
        # select corresponding coll_tl_df
        collection = seg_rec.collection
        coll_idx = self.collections.index(collection)
        coll_tl_df = self.list_coll_tl_df[coll_idx]
        # check if transliterations available
        if len(coll_tl_df) > 0:
            # select corresponding tl_df slice in coll_df
            tl_df = coll_tl_df[coll_tl_df.segm_idx == seg_rec.name]
            # compute number lines
            num_lines = tl_df.line_idx.nunique()
        # report if transliteration is missing
        if len(tl_df) == 0:
            if verbose:
                print('No transliteration found for {}!'.format(seg_rec.tablet_CDLI))
        return tl_df, num_lines



