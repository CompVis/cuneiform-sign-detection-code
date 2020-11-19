import pandas as pd


def load_cunei_mzl_df(path_to_csv='./cunei_mzl.csv', filter=False):
    cunei_mzl_df = pd.read_csv(path_to_csv, index_col=0)
    # avoid mzl idx without codepoint
    cunei_mzl_df = cunei_mzl_df[cunei_mzl_df.num_cpts > 0]
    # deal with multiple versions
    #cunei_mzl_df = cunei_mzl_df.groupby('MesZL', sort=False, as_index=False).first()
    # create composite sign
    cunei_mzl_df['comp_script'] = cunei_mzl_df[['script_0', 'script_1', 'script_2']].fillna('').apply(
        lambda x: ''.join(x), axis=1)
    # decode to unicode (for matching with oracc utf8)
    cunei_mzl_df.comp_script = cunei_mzl_df.comp_script.apply(lambda x: x.decode('utf8'))

    if filter:
        # avoid mzl idx without codepoint
        cunei_mzl_df = cunei_mzl_df[cunei_mzl_df.num_cpts > 0]
        # deal with multiple versions
        cunei_mzl_df = cunei_mzl_df.groupby('MesZL', sort=False, as_index=False).first()

    return cunei_mzl_df


# def get_unicode(mzl_idx, cunei_mzl_df):
#     select_mzl_idx = cunei_mzl_df.MesZL.isin([mzl_idx])
#     if select_mzl_idx.any():
#         cpt_hex = cunei_mzl_df.codepoint_0[select_mzl_idx].str[2:].values[0]  # get hex
#         cpt_int = int(cpt_hex, 16)  # convert to int
#         return unichr(cpt_int)
#     else:
#         return mzl_idx


def get_unicode_comp(mzl_idx, cunei_mzl_df):
    # also handle composite signs by concatenation
    select_mzl_idx = cunei_mzl_df.MesZL.isin([mzl_idx])
    if select_mzl_idx.any():
        cunei_rec = cunei_mzl_df[select_mzl_idx]
        out_str = ''
        for i in range(cunei_rec.num_cpts):
            cpt_hex = cunei_rec['codepoint_{}'.format(i)].str[2:].values[0]  # get hex
            cpt_int = int(cpt_hex, 16)  # convert to int
            out_str += unichr(cpt_int)
        return out_str
    else:
        return mzl_idx


def get_sign_name(mzl_idx, cunei_mzl_df):
    select_mzl_idx = cunei_mzl_df.MesZL.isin([mzl_idx])
    if select_mzl_idx.any():
        cunei_rec = cunei_mzl_df[select_mzl_idx]
        #return cunei_rec['Sign Name'].str.decode('utf8').item()
        return cunei_rec['Sign Name'].str.decode('utf8').str.split('(').str[0].item()
    else:
        return mzl_idx
