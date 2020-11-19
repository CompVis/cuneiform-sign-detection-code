# -*- coding: utf-8 -*-
"""
@author: tdencker
"""

import pandas as pd


class SignsStats(object):

    def __init__(self, tblSignHeight=128, stats_csv_file='../../data/unicode_sign_stats.csv'):
        self.tblSignHeight = tblSignHeight
        self.sign_df = None
        # load stats file
        self.load_stats_from_file(stats_csv_file)

    def load_stats_from_file(self, stats_csv_file):
        # Load sign stats
        sign_df = pd.read_csv(stats_csv_file)
        sign_df = sign_df.set_index('train_lbl')
        # assign
        self.sign_df = sign_df

    def get_sign_width(self, train_lbl, sign_width=None):
        """ Return width of sign from stats """
        # check default sign width
        if sign_width is None:
            sign_width = self.tblSignHeight
        if train_lbl in self.sign_df.index:
            sign_width = self.sign_df.width.loc[train_lbl]
        return sign_width

