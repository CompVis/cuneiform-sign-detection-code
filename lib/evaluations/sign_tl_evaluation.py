import numpy as np
import pandas as pd

import editdistance
import Levenshtein
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

from .sign_evaluation import evaluate_on_gt

# deprecated (should be handled by sign_evaluator)
def get_eval_stats(gt_boxes, gt_labels, aligned_list):
    # evaluate
    num_imgs = 1
    all_tp, all_fp, det_stats, total_num_tp = evaluate_on_gt(gt_boxes, gt_labels,
                                                             num_imgs, [[el] for el in aligned_list])

    total_num_fp = int(np.sum(np.array(det_stats)[:, 3]))
    # print stats
    pd.set_option('display.max_rows', 50)
    df_stats = pd.DataFrame(det_stats, columns=['num_gt', 'num_det', 'tp', 'fp', 'ap', 'lbl'])

    print("total_tp", total_num_tp, "total_fp", total_num_fp,
          # here precision = accuarcy
          "acc", '{:0.2f}'.format(total_num_tp / float(total_num_tp + total_num_fp)),
          "mAP", '{:0.4f}'.format(df_stats['ap'].mean()),
          "mAP(nonzero)", '{:0.4f}'.format(df_stats['ap'].iloc[df_stats['ap'].nonzero()[0]].mean()))
    acc = total_num_tp / float(total_num_tp + total_num_fp)

    return acc, df_stats

# deprecated (should be handled by sign_evaluator)
def compute_accuracy(gt_boxes, gt_labels, aligned_list, return_stats=False):
    # only run if gt available
    if len(gt_boxes) > 0:
        acc, df_stats = get_eval_stats(gt_boxes, gt_labels, aligned_list)
        if return_stats:
            return acc, df_stats
        else:
            return acc
    else:
        return -1


def convert_alignments_for_eval(detections, total_labels=240):
    # convert from RANSAC format (Nx9) to all_boxes
    all_boxes = [[] for _ in range(total_labels)]

    for temp in detections:
        # temp: [ID, cx, cy, score, x1, y1, x2, y2, idx]

        # copy data to _new_ all_boxes
        box = np.zeros((1, 5))
        box[0, :4] = temp[4:8]
        box[0, 4] = temp[3]
        all_boxes[np.int(temp[0])].append(box)

    # for each class stack list of bounding boxes together
    all_boxes = [np.stack(el).squeeze(axis=1) if len(el) > 0 else el for el in all_boxes]

    return all_boxes


#  SCORE FUNCTIONS


def compute_bleu_score(candidate_words, reference_words):
    reference = [reference_words]
    candidate = candidate_words
    # compute score

    # deal with issue
    # https://github.com/nltk/nltk/issues/1554
    hyp_lengths = len(reference_words)
    weights = (0.25, 0.25, 0.25, 0.25)
    if hyp_lengths < 4:
        if hyp_lengths == 0:
            weights = (0, )
        else:
            weights = (1 / float(hyp_lengths), ) * hyp_lengths

    chencherry = SmoothingFunction()
    score = sentence_bleu(reference, candidate, weights=weights, smoothing_function=chencherry.method1)
    return score


def compute_levenshtein(candidate, reference, normalize=True):
    edist = 0
    if len(reference) > 0:
        # strict normalization in [0,1] range
        edist = editdistance.eval(reference, candidate)
        if normalize:
            edist = float(edist) / max(len(reference), len(candidate))
    return edist


def compute_cer(candidate, reference):
    # character error rate (see also WER)
    # character accuracy 1 - CER
    edist = 0
    if len(reference) > 0:
        edist = editdistance.eval(reference, candidate)
        edist = float(edist) / len(reference)
    return edist


def compute_levenshtein_ops(candidate, reference, normalize=True):
    # https://rawgit.com/ztane/python-Levenshtein/master/docs/Levenshtein.html
    ops_dict = {'insert': 0, 'delete': 1, 'replace': 2}
    # print candidate, reference
    edist = 0
    edit_ops = np.zeros(len(ops_dict))
    if len(reference) > 0:
        # convert to string for Levenshtein function
        candidate_str = u''.join([unichr(lbl) for lbl in candidate])
        reference_str = u''.join([unichr(lbl) for lbl in reference])
        # compute ed ops
        ops_df = pd.DataFrame(Levenshtein.editops(candidate_str, reference_str), columns=['type', 'ixA', 'ixB'])
        edist = len(ops_df)
        # collect types
        for op, ii in ops_dict.iteritems():
            edit_ops[ii] = len(ops_df[ops_df.type == op])
        if normalize:
            edist = float(edist) / max(len(reference), len(candidate))

    return edist, edit_ops



