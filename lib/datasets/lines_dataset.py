import os
import numpy as np
import pandas as pd
from PIL import Image
from ast import literal_eval

from scipy import ndimage as ndi
from skimage.util import invert
from skimage.draw import line, line_aa

import torch.utils.data as data
from tqdm import tqdm


from ..utils.bbox_utils import clip_boxes
from ..utils.transform_utils import crop_pil_image
from ..detection.sign_detection import *


### helper functions

def collect_line_coords(seg_line_df, scale=1):
    # group according to line idx
    grouped = seg_line_df.groupby('line_idx')

    # collect all line coordinates
    rr_list, cc_list, lbbox_list = [], [], []
    for i, line_rec in grouped:
        xx = np.rint(line_rec.x.values * scale).astype(int)
        yy = np.rint(line_rec.y.values * scale).astype(int)
        lbbox = np.array([np.min(xx), np.min(yy), np.max(xx), np.max(yy)])
        lbbox_list.append(lbbox)
        for li in range(len(xx) - 1):
            rr, cc, _ = line_aa(yy[li], xx[li], yy[li + 1], xx[li + 1])
            # rr, cc = line(yy[li], xx[li], yy[li+1], xx[li+1])
            rr_list.append(rr)
            cc_list.append(cc)

    # stack coordinates
    rr = np.hstack(rr_list)
    cc = np.hstack(cc_list)
    lbboxes = np.stack(lbbox_list)
    return rr, cc, lbboxes


def create_line_trafo(rr, cc, input_shape):
    # create mask
    line_mask = np.zeros(input_shape).astype(bool)
    line_mask[rr, cc] = 1
    # compute distance transform after inverting
    line_trafo = ndi.distance_transform_edt(invert(line_mask))
    return line_trafo


def compute_sampling_freq(line_trafo, sample_mask, sample_radius, expo=2):
    sample_freq = line_trafo
    # convert to probs
    sample_freq = (-sample_freq / sample_radius + 1) ** expo
    # sample_freq = -sample_freq/sample_radius + 1
    # sample_freq = np.exp(-sample_freq/sample_radius * 2)

    # set area that is not sampled from to 'zero'
    sample_freq[sample_mask < 1] = 0
    return sample_freq


def spatial_sample(sample_freq):
    thresh = np.random.random_sample()
    ylist, xlist = np.where(sample_freq > thresh)
    select_idx = np.random.randint(len(xlist))
    return xlist[select_idx], ylist[select_idx]


def spatial_sample_negative(sample_freq):
    # too slow
    if 0:
        # remove samples close to border
        border_mask = np.zeros_like(sample_freq, dtype=bool)
        bdist = 150
        border_mask[bdist:-bdist, bdist:-bdist] = True
        # apply masks
        ylist, xlist = np.where((sample_freq == 0) & (border_mask))
        select_idx = np.random.randint(len(xlist))
        return xlist[select_idx], ylist[select_idx]
    # faster
    if 1:
        # remove samples close to border
        border_mask = np.zeros_like(sample_freq, dtype=bool)
        bdist = 150
        border_mask[bdist:-bdist, bdist:-bdist] = True
        x, y = 0, 0
        # (line_map[x, y] is True) results in overlap with hard negative samples
        for i in range(100):
            # pick coordinate
            select_idx = np.random.randint(np.prod(sample_freq.shape))
            # back to matrix index
            x, y = np.unravel_index(select_idx, sample_freq.shape)
            if (sample_freq[x, y] == 0) and (border_mask[x, y] == True):
                break
        return y, x


def pad_bboxes(lbboxes, context_pad):
    # works inplace, so need to return
    for bb in lbboxes:
        bb[:2] = bb[:2] - context_pad
        bb[2:4] = bb[2:4] + context_pad
    # return lbboxes


def spatial_sample_line(sample_freq, lbbox):
    thresh = np.random.random_sample()
    ylist, xlist = np.where(sample_freq[lbbox[1]:lbbox[3], lbbox[0]:lbbox[2]] >= thresh)
    if len(xlist) == 0:
        print lbbox, sample_freq.shape
    select_idx = np.random.randint(len(xlist))
    return lbbox[0] + xlist[select_idx], lbbox[1] + ylist[select_idx]


### CuneiformLine Class

class CuneiformLines(data.Dataset):

    def __init__(self, dataset_params, transform=None, target_transform=None, relative_path='../', split='train'):
        # annotation_path, params,

        # set params
        self.line_height = dataset_params['line_height']
        self.sample_radius = dataset_params['sample_radius']  # self.line_height * 3
        self.expo = dataset_params['expo']
        if 'train' in split:
            self.soft_bg_frac = dataset_params['soft_bg_frac'][0]
        else:
            self.soft_bg_frac = dataset_params['soft_bg_frac'][1]

        self.crop_size = dataset_params['crop_size']
        self.patch_size = dataset_params['patch_size']

        # transforms for data preparation
        self.transform = transform
        self.target_transform = target_transform

        # load line annotation
        annotation_file = '{}data/annotations/line_annotations_{}.csv'.format(relative_path, split)
        line_anno_df = pd.read_csv(annotation_file, engine='python')

        # load segment metadata
        annotation_file = '{}data/segments/tablet_segments_{}.csv'.format(relative_path, split)
        tablet_segments_df = pd.read_csv(annotation_file, engine='python', index_col=0)
        # convert string of list to list
        tablet_segments_df['bbox'] = tablet_segments_df['bbox'].apply(literal_eval)
        tablet_segments_df['bbox'] = tablet_segments_df['bbox'].apply(np.array)  # convert to ndarray
        # additional columns
        tablet_segments_df['imageName'] = tablet_segments_df['tablet_CDLI'] + '.jpg'
        tablet_segments_df['im_path'] = '{}data/images/'.format(relative_path) + \
                                        tablet_segments_df['collection'] + '/' + tablet_segments_df['imageName']

        # select assigned
        assigned_segments_df = tablet_segments_df[tablet_segments_df.assigned == True]

        # pre-load segments and compute line and sampling maps
        self.valid_indices = []
        self.num_lines_list = []
        self.image_data_list = []
        self.line_map_list = []
        self.sample_freq_list = []
        lbboxes_list = []
        for segment_idx, segment_rec in tqdm(assigned_segments_df.iterrows(), total=len(assigned_segments_df)):
            imageName = segment_rec.tablet_CDLI
            scale = segment_rec.scale
            seg_bbox = segment_rec.bbox
            path_to_image = segment_rec.im_path
            view_desc = "{}".format(segment_rec.view_desc).replace("nan", "")

            # select line annotations
            seg_line_df = line_anno_df[line_anno_df.segm_idx == segment_idx]

            # check if any annotations available
            if len(seg_line_df) > 0:
                # print(split, imageName, view_desc)

                ### 1) load segment
                # prepare input tablet
                pil_im = Image.open(path_to_image)
                tablet_seg, new_bbox = crop_segment_from_tablet_im(pil_im, seg_bbox)
                # scale image
                input_im = rescale_segment_single(tablet_seg, scale)
                input_shape = input_im.size[::-1]

                ### 2) line map
                # compute interpolated line coordinates

                # collect all line coordinates
                rr, cc, lbboxes = collect_line_coords(seg_line_df, scale=scale)
                # pad with sample radius
                pad_bboxes(lbboxes, self.sample_radius)
                clip_boxes(lbboxes, input_shape)
                # compute line trafo
                line_trafo = create_line_trafo(rr, cc, input_shape)
                # compute masks
                line_mask = line_trafo < self.line_height
                sample_mask = line_trafo < self.sample_radius
                # compute frequency
                sample_freq = compute_sampling_freq(line_trafo, sample_mask, self.sample_radius, self.expo)

                ### 3) save data
                # append to list
                self.valid_indices.append(segment_idx)
                self.num_lines_list.append(len(seg_line_df.line_idx.unique()))
                self.image_data_list.append(input_im)
                self.line_map_list.append(line_mask)
                self.sample_freq_list.append(sample_freq)
                lbboxes_list.append(lbboxes)

        # stack lbboxes
        self.lbboxes = np.vstack(lbboxes_list)

        self.line2mem_list = []
        # for valid_idx, num_lines in zip(self.valid_indices, self.num_lines_list):
        for men_idx, num_lines in enumerate(self.num_lines_list):
            self.line2mem_list.extend([men_idx] * num_lines)

        # Balance sampling with line length
        # 1) get line factors by line width and normalisation
        widths = self.lbboxes[:, 2] - self.lbboxes[:, 0]
        # factor required to make smallest length larger equal 1
        norm_factor_int = np.ceil(float(widths.sum()) / widths.min())
        norm_widths = widths / float(widths.sum())
        line_factors = np.rint(norm_factor_int * norm_widths).astype(int)
        # 2) compute list to sample from
        self.sample2line_list = []
        for ii, line_factor in enumerate(line_factors):
            self.sample2line_list.extend([ii] * line_factor)

        # increase test set size to obtain more stable error
        if split == 'test':
            self.sample2line_list = self.sample2line_list * 5

        # setup finished
        print("Setup {} dataset with {} rows and {} samples".format(split, len(self.line2mem_list), len(self)))

    def __getitem__(self, index):

        # line_index = index
        line_index = self.sample2line_list[index]

        lbbox = self.lbboxes[line_index]
        mem_idx = self.line2mem_list[line_index]
        # get required data
        segm_im = self.image_data_list[mem_idx]
        line_map = self.line_map_list[mem_idx]
        sample_freq = self.sample_freq_list[mem_idx]

        if np.random.random() > self.soft_bg_frac:
            # sample spatial location
            # y, x = spatial_sample(sample_freq) # coordinates need to be inverted
            y, x = spatial_sample_line(sample_freq, lbbox)
            # compute target label
            target = int(line_map[x, y])
        else:
            y, x = spatial_sample_negative(sample_freq)
            # compute target label
            target = int(line_map[x, y])  # should be always negative

        # crop patch at sampled location (use PIL for that)
        hw, hh = self.patch_size[0] / 2., self.patch_size[1] / 2.
        bbox = [y - hw, x - hh, y + hw, x + hh]

        # new fast
        im, bb = crop_pil_image(segm_im, bbox, context_pad=0, pad_to_square=False)

        # apply augmentation pipeline and convert from PIL to numpy
        if self.transform is not None:
            im = self.transform(im)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return im, target

    def __len__(self):
        # return total lines
        # return len(self.sample_indices)
        return len(self.sample2line_list)


