import torch
import numpy as np
import pandas as pd
from PIL import Image
from ast import literal_eval
import os.path
from tqdm import tqdm

import torch.utils.data as data

from ..detection.sign_detection import crop_segment_from_tablet_im

# from utils.cython_bbox import bbox_overlaps
from ..utils.bbox_utils import clip_boxes
from ..utils.torchcv.transforms.crop_box import crop_box
from ..utils.torchcv.transforms.resize import resize


# helper functions

def convert_bbox_global2local(gbbox, seg_bbox):
    x, y = seg_bbox[:2]
    relative_bbox = np.array(gbbox) - np.array([x, y, x, y])
    return relative_bbox.tolist()


def get_segment_meta(segment_rec):
    image_name = segment_rec.tablet_CDLI

    # this should control which scale is used in consecutive processing
    scale = segment_rec.scale  #* self.rescale

    seg_bbox = segment_rec.bbox
    path_to_image = segment_rec.im_path
    view_desc = "{}".format(segment_rec.view_desc).replace("nan", "")

    return image_name, scale, seg_bbox, path_to_image, view_desc


def bbox_ctr_overlaps(boxes1, boxes2):
    # check for all combinations of boxes1 and boxes2 if ctrs of boxes2 are in boxes1
    overlaps_mat = np.zeros([boxes1.shape[0], boxes2.shape[0]])
    for ii, box in enumerate(boxes1):
        x, y, x2, y2 = box
        # check if center is still inside tile_box, otherwise ignore box
        # if center is not inside tile box,
        # not possible to get IoU >= 0.5 --> treated as background anyways
        center = (boxes2[:, :2] + boxes2[:, 2:]) / 2
        mask = (center[:, 0] >= x) & (center[:, 0] <= x2) \
               & (center[:, 1] >= y) & (center[:, 1] <= y2)
        overlaps_mat[ii, :] = mask
    return overlaps_mat


# Cuneiform SSD dataset


class CuneiformSegments(data.Dataset):

    def __init__(self, collections=['train'], transform=None, relative_path='../',
                 only_annotated=True, only_assigned=True, preload_segments=True, use_gray_scale=True):

        # merge multiple data sources in order to provide following function:
        #  f(idx) -> image, boxes, labels
        # uses gt annotations only
        # if no annotations available boxes and labels are empty lists

        # transforms for data preparation
        self.transform = transform
        self.preload_segments = preload_segments
        self.use_gray_scale = use_gray_scale

        ### load and prepare list_sign_anno_df
        # manual annotation files may be based on multiple collections
        # for each collection
        # store in list_sign_anno_df

        # load bbox annotations
        list_anno_collections = []
        sign_anno_df_list = []
        for collection in collections:
            # load sign annotations
            annotation_file = '{}data/annotations/bbox_annotations_{}.csv'.format(relative_path, collection)
            # ATTENTION: only use gt annotations if collection is provided in collections parameter
            if os.path.exists(annotation_file):
                sign_anno_df = pd.read_csv(annotation_file, engine='python')  # read annotation file
                # add additional columns
                sign_anno_df['generated'] = False
                sign_anno_df['global_segm_idx'] = -1
                sign_anno_df['relative_bbox'] = sign_anno_df['relative_bbox'].apply(literal_eval)
                sign_anno_df['relative_bbox'] = sign_anno_df['relative_bbox'].apply(np.array)  # convert to ndarray

                # slice sign_anno_df if there are multiple different collections contained
                for sub_collection in sign_anno_df.collection.unique():
                    # store collection name
                    list_anno_collections.append(sub_collection)
                    # store collection specific slice of data frame
                    sub_sign_anno_df = sign_anno_df[sign_anno_df.collection == sub_collection]
                    sign_anno_df_list.append(sub_sign_anno_df)

        ### extend collections
        # create list of elementary collections
        collections_ext = np.unique(list_anno_collections).tolist()

        ###################
        # II) on collection level: load annotations and meta data

        ### load segment, sign meta information
        # for each collection
        # store in segments_df_list

        # reduced set of columns - only keep what is needed and maintained
        segments_df_columns = ['tablet_CDLI', 'view_desc', 'bbox', 'collection', 'scale', 'im_path']

        segments_df_list = []
        #sign_anno_df_list = []
        for collection in collections_ext:

            # load segment metadata
            annotation_file = '{}data/segments/tablet_segments_{}.csv'.format(relative_path, collection)
            tablet_segments_df = pd.read_csv(annotation_file, engine='python', index_col=0)
            # convert string of list to list
            tablet_segments_df['bbox'] = tablet_segments_df['bbox'].apply(literal_eval)
            tablet_segments_df['bbox'] = tablet_segments_df['bbox'].apply(np.array)  # convert to ndarray
            # add collection column
            file_names = tablet_segments_df['tablet_CDLI'] + '.jpg'
            tablet_segments_df['im_path'] = '{}data/images/'.format(relative_path) + tablet_segments_df['collection'] + '/' + file_names
            # get assigned segment (can be edited from outside without harm)
            if only_assigned:
                assigned_segments_df = tablet_segments_df[(tablet_segments_df.assigned == True)]
            else:
                assigned_segments_df = tablet_segments_df

            # collect data frames in lists
            segments_df_list.append(assigned_segments_df[segments_df_columns])


        ### assemble ssd_segments_df with new index
        # search all segments with annotations

        list_segments_df_anno = []
        for collection in collections_ext:
            coll_idx = collections_ext.index(collection)

            list_segm_indices = []
            # get all segment indices for this collection that contain annotations
            if only_annotated:
                # if there are gt annotations
                if collection in list_anno_collections:
                    anno_coll_idx = list_anno_collections.index(collection)
                    # if there are gt annotations
                    if len(sign_anno_df_list[anno_coll_idx]) > 0:
                        # load their indices
                        segm_indices_anno = sign_anno_df_list[anno_coll_idx].segm_idx.unique()
                        # filter annotations without assigned segment
                        segm_indices_anno = segm_indices_anno[segm_indices_anno >= 0]
                        list_segm_indices.append(segm_indices_anno)
                # append only segments with anno
                if len(list_segm_indices) > 0:
                    # stack to obtain list of segment indices with annotations
                    segm_indices = np.unique(np.hstack(list_segm_indices))
                    # append
                    list_segments_df_anno.append(segments_df_list[coll_idx].loc[segm_indices])
            else:
                # append all segments from collection
                list_segments_df_anno.append(segments_df_list[coll_idx])

        # create new datasets ssd_segment_df
        # concat dataframes and use reset_index to create column with old indices
        ssd_segments_df = pd.concat(list_segments_df_anno).reset_index()

        # rename column to segm_idx
        ssd_segments_df.columns.values[0] = 'segm_idx'


        ###################
        # III) on segment level: load data and prepare dataset index

        ### assemble ssd_sign_anno_df and update ssd_segments_df
        # additional column for ssd_sign_anno_df: global_segm_idx
        # additional column for ssd_segments_df: with num_anno

        sign_anno_df_cols = ['tablet_CDLI', 'mzl_label', 'train_label', 'segm_idx', 'collection',
                             'generated', 'relative_bbox', 'global_segm_idx']
        # segm_idx,tablet_CDLI,view_desc,collection,mzl_label,train_label,bbox,relative_bbox
        list_ssd_sign_anno_df = []

        list_lines_annotated_per_segm = np.zeros(len(ssd_segments_df), dtype=bool)
        list_num_anno_per_segm = np.zeros(len(ssd_segments_df), dtype=int)

        # iterate over segments
        for global_seg_idx, seg_rec in ssd_segments_df.iterrows():
            image_name, scale, seg_bbox, image_path, view_desc = get_segment_meta(seg_rec)
            res_name = "{}{}".format(image_name, view_desc)
            segm_idx = seg_rec.segm_idx
            collection = seg_rec.collection
            # print(image_name, view_desc, segm_idx)
            coll_idx = collections_ext.index(collection)

            ### if annotations available for segment, append to list
            if collection in list_anno_collections:
                anno_coll_idx = list_anno_collections.index(collection)
                if len(sign_anno_df_list[anno_coll_idx]) > 0:
                    sign_anno_df = sign_anno_df_list[anno_coll_idx]
                    # select sign annos for segment
                    segm_select = sign_anno_df.segm_idx == segm_idx
                    if len(sign_anno_df[segm_select]) > 0:
                        # update data frame column
                        sign_anno_df.loc[segm_select, 'global_segm_idx'] = global_seg_idx
                        # collect information
                        sign_anno_seg = sign_anno_df[segm_select]
                        list_num_anno_per_segm[global_seg_idx] = len(sign_anno_seg)
                        list_ssd_sign_anno_df.append(sign_anno_seg[sign_anno_df_cols])

        # add columns to ssd_segments_df
        ssd_segments_df['num_anno'] = np.array(list_num_anno_per_segm)

        if len(list_ssd_sign_anno_df) > 0:
            # assemble ssd_sign_anno_df (drop old index)
            ssd_sign_anno_df = pd.concat(list_ssd_sign_anno_df, ignore_index=True)
        else:
            # create empty data frame with correct columns
            ssd_sign_anno_df = pd.DataFrame(columns=sign_anno_df_cols)

        ###################
        # IV) Preload: line detections and segment images

        ### preload segment images
        # crop segment and convert to gray scale
        # IMPORTANT: preload segment crops (without scaling, because memory)

        image_data_list = []
        if self.preload_segments:
            # iterate over segments
            for global_seg_idx, seg_rec in tqdm(ssd_segments_df.iterrows(), total=len(ssd_segments_df)):
                image_name, scale, seg_bbox, image_path, view_desc = get_segment_meta(seg_rec)
                res_name = "{}{}".format(image_name, view_desc)

                # load composite image
                pil_im = Image.open(image_path)
                # crop segment
                tablet_seg, new_bbox = crop_segment_from_tablet_im(pil_im, seg_bbox)
                # convert to gray scale and store in list
                if self.use_gray_scale:
                    # convert to gray scale
                    image_data_list.append(tablet_seg.convert('L'))
                else:
                    image_data_list.append(tablet_seg)


        ###################
        # VI) Dataset index

        sample2tile_list = ssd_segments_df.index.values

        ###################
        # attach resulting data structures to class
        self.collections = collections
        self.collections_ext = collections_ext

        self.ssd_segments_df = ssd_segments_df
        self.ssd_sign_anno_df = ssd_sign_anno_df

        self.image_data_list = image_data_list

        # self.sign_anno_df_list = sign_anno_df_list
        # self.segments_df_list = segments_df_list

        self.sample2tile_list = sample2tile_list

        # map from seg idx to dataset idx
        self.sidx2didx = dict(zip(ssd_segments_df.segm_idx.values, range(len(ssd_segments_df))))

        # setup finished
        print("Setup dataset spanning {} collections with {} annotations [{} segments, {} indices]".format(
            len(collections_ext), len(ssd_sign_anno_df), len(ssd_segments_df),  len(sample2tile_list)))

    def __getitem__(self, index):
        # get segment
        global_seg_idx = self.sample2tile_list[index]
        seg_rec = self.ssd_segments_df.loc[global_seg_idx]

        # load segment meta data
        image_name, scale, seg_bbox, image_path, view_desc = get_segment_meta(seg_rec)

        # get sign annos
        select_segm = self.ssd_sign_anno_df.global_segm_idx == global_seg_idx
        segm_annos = self.ssd_sign_anno_df[select_segm]

        # get annotated boxes and their labels
        if len(segm_annos) > 0:
            seg_boxes = np.stack(segm_annos.relative_bbox)
            labels = segm_annos.train_label.values
            # convert to torch tensors
            seg_boxes = torch.from_numpy(seg_boxes).float()
            labels = torch.from_numpy(labels)
        else:
            seg_boxes = None
            labels = None

        # get segment image
        if self.preload_segments:
            pil_im = self.image_data_list[global_seg_idx]
        else:
            # load composite image
            pil_im = Image.open(image_path)
            # crop segment
            tablet_seg, new_bbox = crop_segment_from_tablet_im(pil_im, seg_bbox)
            if self.use_gray_scale:
                # convert to gray scale
                pil_im = tablet_seg.convert('L')
            else:
                pil_im = tablet_seg

        # tensor functions adapted from kuangliu's code
        # https://github.com/kuangliu/torchcv/tree/master/torchcv/transforms

        # scale segment
        im, boxes = resize(pil_im, seg_boxes, None, scale=scale)

        # apply augmentation pipeline and convert from PIL to numpy
        if self.transform is not None:
            im, boxes, labels = self.transform(im, boxes, labels)

        return im, boxes, labels

    def get_seg_rec(self, index):
        # get segment
        global_seg_idx = self.sample2tile_list[index]
        return self.ssd_segments_df.loc[global_seg_idx]

    def __len__(self):
        return len(self.sample2tile_list)

