# evaluate line-tl alignment using gt-line annotations
# only quality indicator because transliterations are unreliable
# (transliterations often miss lines or contain invisible lines)


def eval_line_tl_alignment(line_frag, lines_anno, seg_idx, num_vis_lines):
    tl_asst_eval = line_frag.line_hypos_agg[['gt_line_idx', 'tl_line']].dropna()
    tl_asst_eval = tl_asst_eval[tl_asst_eval.tl_line >= 0]  # do not count unassigned lines
    print('LineHypos-TL assignment accuracy: {}'.format(
        (tl_asst_eval.gt_line_idx == tl_asst_eval.tl_line).mean()))
    # check if consistent gt and tl
    num_lines_tl = num_vis_lines  # line_frag.tl_df.line_idx.nunique()
    num_lines_gt = lines_anno.select_df_by_segm_idx(seg_idx).gt_line_idx.nunique()
    if num_lines_tl != num_lines_gt:
        print('line annotation - transliteration mismatch: {} vs {} lines  '.format(num_lines_gt, num_lines_tl))
