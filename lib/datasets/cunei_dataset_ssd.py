import numpy as np
import pandas as pd
from PIL import Image
from ast import literal_eval
import os.path
from tqdm import tqdm

import torch.utils.data as data

from ..detection.sign_detection import *

# from utils.cython_bbox import bbox_overlaps
from ..utils.bbox_utils import clip_boxes
from ..utils.transform_utils import convert2binaryPIL
from ..utils.torchcv.transforms.crop_box import crop_box
from ..utils.torchcv.transforms.resize import resize
from ..utils.torchcv.transforms_lm.crop_box import crop_box_lm
from ..utils.torchcv.transforms_lm.resize import resize_lm

from .lines_dataset import collect_line_coords, create_line_trafo

from ..detection.line_detection import compute_image_label_map


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


def compute_tiles(imw, imh, scale, tile_shape=[600, 600], border_sz=100, w_step_sz=300, h_step_sz=400):
    # TODO: improve using linespace and allow overlap to vary

    # signs height should be around 130px, however, length can be up to 300px
    # -> overlap along lines (300px) should be larger than between lines (200px)
    # -> this means for step sizes: w_step_sz < h_step_sz
    inv_scale = 1. / scale
    tile_shape = np.array(tile_shape) * inv_scale
    border_sz *= inv_scale
    w_step_sz *= inv_scale
    h_step_sz *= inv_scale

    tile_ol_w = tile_shape[0] - w_step_sz
    tile_ol_h = tile_shape[0] - h_step_sz
    w_list = np.arange(border_sz, imw - border_sz - tile_ol_w, step=w_step_sz)
    h_list = np.arange(border_sz, imh - border_sz - tile_ol_h, step=h_step_sz)

    # grid pts represent upper left corner of tile box
    # tiles can be larger than image and need to be padded
    XX, YY = np.meshgrid(w_list, h_list)

    # compute bboxes
    ul_corner = np.rint(np.stack([XX.ravel(), YY.ravel()], axis=1)).astype(int)
    lr_corner = ul_corner + np.rint(tile_shape)
    bboxes = np.hstack([ul_corner, lr_corner])
    # make sure tiles inside image boundaries
    bboxes = clip_boxes(bboxes, [imh, imw])  # [imh, imw] is correct order for this function

    return bboxes, XX, YY


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


class CuneiformSSD(data.Dataset):

    def __init__(self, collections=['train'], gen_file_path=None, gen_collections=[], gen_folder=None, transform=None,
                 relative_path='../', use_balanced_idx=True, tile_shape=[600, 600], use_linemaps=False,
                 remove_empty_tiles=False, min_align_ratio=0.6, filter_nms=False, compl_thresh=-1, ncompl_thresh=-1,
                 num_top_ncompl=0, min_ncompl_thresh=10):

        # merge multiple data sources in order to form a single dataset that can be used for SSD style detector training
        # provides following function:
        #  f(idx) -> image, bboxes, labels
        # or more general:
        #  f(idx) -> image, bboxes, labels, line_map

        # join multiple levels of supervision: three cases for sign annotations
        # 1) tablets completely annotated (no need to load line annotations nor line detections)
        # 2) tablets partly annotated and line annotations available (no need to load line detections)
        # 3) tablets partly annotated and line detections required

        # transforms for data preparation
        self.transform = transform
        self.line_model_version = None
        self.use_linemaps = use_linemaps
        self.min_align_ratio = min_align_ratio
        self.filter_nms = filter_nms
        self.compl_thresh = compl_thresh
        self.ncompl_thresh = ncompl_thresh
        self.num_top_ncompl = num_top_ncompl
        self.min_ncompl_thresh = min_ncompl_thresh

        line_model_version = 'v007'
        num_classes = 240

        ###################
        # I) load generated and manual annotations

        ### load and prepare gen_df
        # generated annotations may be based on multiple collections

        gen_cols = ['imageName', 'folder', 'image_path', 'label', 'train_label',
                    'x1', 'y1', 'x2', 'y2', 'width', 'height', 'segm_idx',
                    'line_idx', 'pos_idx', 'det_score', 'm_score', 'align_ratio', 'nms_keep', 'compl', 'ncompl']

        # OPT I : use csv file that contains list of generated boxes
        if gen_file_path:
            gen_file_path = "{}results{}".format(relative_path, gen_file_path)
            gen_df = pd.read_csv(gen_file_path, engine='python', header=None, names=gen_cols)
        # OPT II : load csv files for collection specific collections and concatenate
        elif len(gen_collections) > 0:
            assert gen_folder is not None, 'When using gen_collections, user needs to provide gen_model!'
            df_list = []
            for gen_coll in gen_collections:
                gen_file_path = "{}results/{}line_generated_bboxes_refined80_{}.csv".format(relative_path, gen_folder, gen_coll)
                # special delimiter because of legacy support, thanks to regex possible to support new and old formats
                gen_df = pd.read_csv(gen_file_path, engine='python', delimiter=',\s*', header=None, names=gen_cols)  #delimiter=', ',
                df_list.append(gen_df)
            gen_df = pd.concat(df_list, ignore_index=True)

        # prepare gen_df
        list_gen_collection = []
        if gen_file_path or (len(gen_collections) > 0):

            num_before_filter = len(gen_df)
            # IMPORTANT: filter gen data according to align ratio
            gen_df = gen_df[gen_df.align_ratio > self.min_align_ratio]
            print('Align Ratio {} :: Removed {} samples. [{}]'.format(self.min_align_ratio, num_before_filter - len(gen_df), len(gen_df)))
            num_before_filter = len(gen_df)
            # only keep inlier classes [0-240] (only required when using null hypos)
            gen_df = gen_df[gen_df.train_label < num_classes]
            print('Class Range {} :: Removed {} samples. [{}]'.format(num_classes, num_before_filter - len(gen_df), len(gen_df)))

            # IMPORTANT: fill nan values in a way that avoids filtering
            gen_df.nms_keep = gen_df.nms_keep.fillna(1).astype(bool)
            gen_df.compl = gen_df.compl.fillna(50)
            gen_df.ncompl = gen_df.ncompl.fillna(100)

            num_before_filter = len(gen_df)
            if self.filter_nms:
                # filter using nms
                gen_df = gen_df[gen_df.nms_keep]
                print('NMS :: Removed {} samples. [{}]'.format(num_before_filter - len(gen_df), len(gen_df)))
            num_before_filter = len(gen_df)

            select_topn = False
            if self.num_top_ncompl > 0:
                # find top 5 for each class with more relaxed ncompl condition
                select_min_ncompl = (gen_df.ncompl > self.min_ncompl_thresh)  # necessary condition
                index_list = gen_df[select_min_ncompl].groupby('train_label').ncompl.nlargest(self.num_top_ncompl).index.values
                select_topn = gen_df.index.isin(np.stack(index_list)[:, 1])

            if self.compl_thresh > -1:
                # filter using compl
                gen_df = gen_df[gen_df.compl > self.compl_thresh]   # 0, 2, 4, 5
                print('Completeness {} :: Removed {} samples. [{}]'.format(self.compl_thresh, num_before_filter - len(gen_df), len(gen_df)))
            elif self.ncompl_thresh > -1:
                # filter using compl
                gen_df = gen_df[(gen_df.ncompl > self.ncompl_thresh) | select_topn]   # 0, 2, 4, 5
                print('Completeness (norm.) {} :: Removed {} samples. [{}]'.format(self.ncompl_thresh, num_before_filter - len(gen_df), len(gen_df)))
            print('class sample count stats: ')
            print(gen_df.train_label.value_counts().describe())

            # add additional columns
            gen_df['collection'] = gen_df.folder.str.split('/').str[0]
            gen_df['generated'] = True
            gen_df['global_segm_idx'] = -1
            gen_df['relative_bbox'] = gen_df[['x1', 'y1', 'x2', 'y2']].values.tolist()
            gen_df['relative_bbox'] = gen_df['relative_bbox'].apply(np.array)
            gen_df['mzl_label'] = gen_df['label']
            gen_df['tablet_CDLI'] = gen_df['imageName']

            # identify all collections with generated annotations
            list_gen_collection = gen_df.collection.unique().tolist()


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

                # only keep inlier classes [0-240]
                class_outlier_select = sign_anno_df.train_label < num_classes
                if np.any(class_outlier_select):
                    print('Drop {} outlier class samples from {}!'.format(np.sum(~class_outlier_select), collection))
                    sign_anno_df = sign_anno_df[class_outlier_select]
                # slice sign_anno_df if there are multiple different collections contained
                for sub_collection in sign_anno_df.collection.unique():
                    # store collection name
                    list_anno_collections.append(sub_collection)
                    # store collection specific slice of data frame
                    sub_sign_anno_df = sign_anno_df[sign_anno_df.collection == sub_collection]
                    sign_anno_df_list.append(sub_sign_anno_df)


        ### extend collections
        # create list of elementary collections
        collections_ext = np.unique(list_gen_collection + list_anno_collections).tolist()
        #collections_ext

        ###################
        # II) on collection level: load segments meta data and line annotation (optional)

        ### load segment, line
        # for each collection
        # store in segments_df_list, line_anno_df_list

        # reduced set of columns - only keep what is needed and maintained
        # segments_df_columns = ['tablet_CDLI', 'view_desc', 'padded_bbox', 'collection', 'line_scale', 'scale',
        #                        'im_path',
        #                        'num_dets_hd', 'num_signs_visible']

        segments_df_columns = ['tablet_CDLI', 'view_desc', 'bbox', 'collection', 'scale', 'im_path']

        segments_df_list = []
        line_anno_df_list = []
        for collection in collections_ext:

            # load segment metadata
            annotation_file = '{}data/segments/tablet_segments_{}.csv'.format(relative_path, collection)
            tablet_segments_df = pd.read_csv(annotation_file, engine='python', index_col=0)
            # convert string of list to list
            tablet_segments_df['bbox'] = tablet_segments_df['bbox'].apply(literal_eval)
            tablet_segments_df['bbox'] = tablet_segments_df['bbox'].apply(np.array)  # convert to ndarray
            # add additional columns
            tablet_segments_df['imageName'] = tablet_segments_df['tablet_CDLI'] + '.jpg'
            tablet_segments_df['im_path'] = '{}data/images/'.format(relative_path) + \
                                            tablet_segments_df['collection'] + '/' + tablet_segments_df['imageName']
            # get assigned segment (can be edited from outside without harm)
            assigned_segments_df = tablet_segments_df[tablet_segments_df.assigned == True]

            # load line annotations
            annotation_file = '{}data/annotations/line_annotations_{}.csv'.format(relative_path, collection)
            if os.path.exists(annotation_file):
                line_anno_df = pd.read_csv(annotation_file, engine='python')
            else:
                line_anno_df = []

            # collect data frames in lists
            segments_df_list.append(assigned_segments_df[segments_df_columns])
            line_anno_df_list.append(line_anno_df)


        ### assemble ssd_segments_df with new index
        # search all segments with annotations

        list_segments_df_anno = []
        for collection in collections_ext:
            coll_idx = collections_ext.index(collection)
            #print(collection)

            # get all segment indices for this collection that contain annotations
            list_segm_indices = []

            # if there are gt annotations
            if collection in list_anno_collections:
                anno_coll_idx = list_anno_collections.index(collection)
                if len(sign_anno_df_list[anno_coll_idx]) > 0:
                    # load their indices
                    segm_indices_anno = sign_anno_df_list[anno_coll_idx].segm_idx.unique()
                    # filter annotations without assigned segment
                    segm_indices_anno = segm_indices_anno[segm_indices_anno >= 0]
                    list_segm_indices.append(segm_indices_anno)

            # if there are generated annotations
            if collection in list_gen_collection:
                # select gen annotations by collection
                col_gen_df = gen_df[gen_df.collection == collection]
                # load their indices
                segm_indices_anno = col_gen_df.segm_idx.unique()
                list_segm_indices.append(segm_indices_anno)

            # stack to obtain list of segment indices with annotations
            segm_indices = np.unique(np.hstack(list_segm_indices))

            # append only segments with anno
            if len(segm_indices) > 0:
                list_segments_df_anno.append(segments_df_list[coll_idx].loc[segm_indices])

        # create new datasets ssd_segment_df
        # concat dataframes and use reset_index to create column with old indices
        ssd_segments_df = pd.concat(list_segments_df_anno).reset_index()
        # rename column to segm_idx
        ssd_segments_df.columns.values[0] = 'segm_idx'


        ###################
        # III) on segment level: load data and prepare dataset index

        ### assemble ssd_sign_anno_df and update ssd_segments_df
        # make sure all annos have relative_bbox
        # additional column for ssd_sign_anno_df: global_segm_idx
        # add two columns to ssd_segments_df: with num_anno, with_line_anno
        # type of annotation: full, partly_w_line_anno, partly_w_line_dect

        # sign_anno_df_cols = ['imageName', 'image_path', 'label', 'train_label', 'segm_idx', 'collection',
        #                      'generated', 'relative_bbox', 'global_segm_idx']
        sign_anno_df_cols = ['tablet_CDLI', 'mzl_label', 'train_label', 'segm_idx', 'collection',
                             'generated', 'relative_bbox', 'global_segm_idx']
        list_ssd_sign_anno_df = []

        list_lines_annotated_per_segm = np.zeros(len(ssd_segments_df), dtype=bool)
        list_num_anno_per_segm = np.zeros(len(ssd_segments_df), dtype=int)

        # iterate over segments
        for global_seg_idx, seg_rec in ssd_segments_df.iterrows():
            image_name, scale, seg_bbox, image_path, view_desc = get_segment_meta(seg_rec)
            res_name = "{}{}".format(image_name, view_desc)
            collection = seg_rec.collection
            segm_idx = seg_rec.segm_idx
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

            ### if generated annotations available, append to list
            if collection in list_gen_collection:
                # select sign annos for segment AND collection
                segm_select = (gen_df.segm_idx == segm_idx) & (gen_df.collection == seg_rec.collection)
                if len(gen_df[segm_select]) > 0:
                    # update data frame columns
                    gen_df.loc[segm_select, 'global_segm_idx'] = global_seg_idx
                    # compute relative_bbox
                    relative_boxes = gen_df[segm_select].relative_bbox.apply(
                        lambda x: np.rint(convert_bbox_global2local(x, list(seg_bbox))).astype(int))
                    gen_df.loc[segm_select, 'relative_bbox'] = relative_boxes

                    # collect information
                    sign_anno_seg = gen_df[segm_select]
                    list_num_anno_per_segm[global_seg_idx] = len(sign_anno_seg)
                    list_ssd_sign_anno_df.append(sign_anno_seg[sign_anno_df_cols])

            ### check for line annotations
            if len(line_anno_df_list[coll_idx]) > 0:
                line_anno_df = line_anno_df_list[coll_idx]
                # select line annos for segment
                segm_select = line_anno_df.segm_idx == segm_idx
                # if there are line annotations for segment
                if len(line_anno_df[segm_select]) > 0:
                    # assume all lines are annotated and remember type of line data
                    list_lines_annotated_per_segm[global_seg_idx] = True

        # add columns to ssd_segments_df
        ssd_segments_df['num_anno'] = np.array(list_num_anno_per_segm)
        ssd_segments_df['with_line_anno'] = list_lines_annotated_per_segm

        # assemble ssd_sign_anno_df (drop old index)
        ssd_sign_anno_df = pd.concat(list_ssd_sign_anno_df, ignore_index=True)

        # this is deprecated, since bug fix
        #assert np.sum(ssd_sign_anno_df.groupby('global_segm_idx').collection.nunique() > 1) == 0

        ###################
        # IV) Preload: segment images and line detections


        ### preload segment images
        # crop segment and convert to gray scale
        # IMPORTANT: preload segment crops (without scaling, because memory)

        image_data_list = []

        # iterate over segments
        for global_seg_idx, seg_rec in tqdm(ssd_segments_df.iterrows(), total=len(ssd_segments_df)):
            image_name, scale, seg_bbox, image_path, view_desc = get_segment_meta(seg_rec)
            res_name = "{}{}".format(image_name, view_desc)

            # load composite image
            pil_im = Image.open(image_path)
            # crop segment
            tablet_seg, new_bbox = crop_segment_from_tablet_im(pil_im, seg_bbox)
            # convert to gray scale and store in list
            image_data_list.append(tablet_seg.convert('L'))



        ### preload line detections
        # could pre-compute line annotations->line map
        # this is a speed memory trade-off

        line_detection_dict = {}
        line_map_dict = {}

        # only required if there are any generated detections
        if self.use_linemaps:

            # iterate over segments
            for global_seg_idx, seg_rec in tqdm(ssd_segments_df.iterrows(), total=len(ssd_segments_df)):
                image_name, scale, seg_bbox, image_path, view_desc = get_segment_meta(seg_rec)
                res_name = "{}{}".format(image_name, view_desc)
                # get collection idx
                coll_idx = collections_ext.index(seg_rec.collection)
                # get seg image shape
                input_shape = np.array(image_data_list[global_seg_idx].size[::-1])

                # if annotations are generated, need to create line map
                #if seg_rec.collection in list_gen_collection:

                # if no line annotations available
                if True:  # ALWAYS use generated annotations not seg_rec.with_line_anno:  # if seg_rec.collection != 'train'
                    # either skeleton or lbl_ind
                    line_res_path = "{}results/results_line/{}/{}".format(relative_path, line_model_version, seg_rec.collection)
                    lines_file = "{}/{}_lbl_ind.npy".format(line_res_path, res_name)
                    # lines_file = "{}/{}_skeleton.npy".format(line_res_path, res_name)
                    lbl_ind_x = np.load(lines_file).astype(int)
                    # store in dictionary
                    line_detection_dict[global_seg_idx] = lbl_ind_x

                    # create line map from detections -> PIL binary
                    lbl_im = create_line_map_from_line_det(line_detection_dict, global_seg_idx, scale, input_shape)

                else:
                    # create line map from line annotations -> PIL binary
                    lbl_im = create_line_map_from_line_anno(line_anno_df_list, coll_idx, seg_rec.segm_idx, input_shape)

                # resize to image size (do here or in next iter
                # lbl_im = lbl_im.resize(input_shape[::-1])

                # store in dictionary
                line_map_dict[global_seg_idx] = lbl_im


        ###################
        # V) Tiling

        ### compute ssd_tile_df
        list_tile_boxes = []
        list_tile_support = []
        list_tile_seg_idx = []

        # iterate over segments
        for global_seg_idx, seg_rec in tqdm(ssd_segments_df.iterrows(), total=len(ssd_segments_df)):
            image_name, scale, seg_bbox, image_path, view_desc = get_segment_meta(seg_rec)
            res_name = "{}{}".format(image_name, view_desc)

            ## compute tiles
            # get segment shape
            imw, imh = image_data_list[global_seg_idx].size
            # compute tile boxes
            tile_boxes, _, _ = compute_tiles(imw, imh, scale, tile_shape=tile_shape)
            # append
            list_tile_boxes.append(tile_boxes)
            list_tile_seg_idx.append([global_seg_idx] * len(tile_boxes))

            ## check overlap of tile boxes and sign boxes
            # get annotations
            seg_sign_annos = ssd_sign_anno_df[ssd_sign_anno_df.global_segm_idx == global_seg_idx]
            sign_bboxes = np.stack(seg_sign_annos.relative_bbox.values)

            # OPT I: compute IOU
            # tiles_sign_iou = bbox_overlaps(tile_boxes.astype(float), sign_bboxes.astype(float))
            # tile_support = np.sum(tiles_sign_iou > 0.005, axis=1)  # 0.01 or 0.005

            # OPT II: compute ctr overlap (strict)
            tiles_sign_ctrs = bbox_ctr_overlaps(tile_boxes.astype(float), sign_bboxes.astype(float))
            tile_support = np.sum(tiles_sign_ctrs, axis=1).astype(int)
            list_tile_support.append(tile_support)

        # stack tile boxes
        tile_boxes_arr = np.vstack(list_tile_boxes)
        tile_global_seg_idx = np.hstack(list_tile_seg_idx).astype(int)
        tile_support_arr = np.hstack(list_tile_support)

        # create tile_df
        tile_df = pd.DataFrame({'global_segm_idx': tile_global_seg_idx,
                                'tile_bbox': tile_boxes_arr.tolist(),
                                'num_anno': tile_support_arr})

        # OPTIONAL: filter tiles with little support
        if remove_empty_tiles and not use_balanced_idx:
            tile_df = tile_df[tile_df.num_anno > 0]    # 0
            tile_df.reset_index(drop=True)

        ###################
        # VI) Dataset index

        ## Balance sampling of tiles with anno per tile
        # create an dataset index which is proportional to annotations per tile
        # attention: tiles without support will be ignored!
        use_balanced_idx = use_balanced_idx    # good for debug

        # 1) get tile factors
        tile_factors = tile_df.num_anno.values
        # 2) compute list to sample from
        if use_balanced_idx:
            sample2tile_list = []
            for ii, tile_factor in enumerate(tile_factors):
                sample2tile_list.extend([ii] * tile_factor)
        else:
            sample2tile_list = tile_df.index.values

        ###################
        # attach resulting data structures to class
        self.collections = collections
        self.collections_ext = collections_ext

        self.ssd_segments_df = ssd_segments_df
        self.ssd_sign_anno_df = ssd_sign_anno_df
        self.tile_df = tile_df

        self.image_data_list = image_data_list
        # self.line_detection_dict = line_detection_dict
        self.line_map_dict = line_map_dict

        self.line_anno_df_list = line_anno_df_list
        # self.sign_anno_df_list = sign_anno_df_list
        # self.segments_df_list = segments_df_list

        self.sample2tile_list = sample2tile_list

        # setup finished
        print("Setup dataset spanning {} collections with {} annotations [{} segments, {} tiles, {} indices]".format(
            len(collections_ext), len(ssd_sign_anno_df), len(ssd_segments_df), len(tile_df), len(sample2tile_list)))

    def __getitem__(self, index):
        # get tile
        tile_index = self.sample2tile_list[index]
        tile_rec = self.tile_df.loc[tile_index]
        tile_bbox = tile_rec.tile_bbox

        # get segment
        global_seg_idx = tile_rec.global_segm_idx
        seg_rec = self.ssd_segments_df.loc[global_seg_idx]
        coll_idx = self.collections_ext.index(seg_rec.collection)

        # load segment meta data
        image_name, scale, seg_bbox, path_to_image, view_desc = get_segment_meta(seg_rec)
        with_line_anno = seg_rec.with_line_anno

        # get segment image
        pil_im = self.image_data_list[global_seg_idx]

        # get sign annos
        select_segm = self.ssd_sign_anno_df.global_segm_idx == global_seg_idx
        segm_annos = self.ssd_sign_anno_df[select_segm]
        seg_boxes = np.stack(segm_annos.relative_bbox)
        labels = segm_annos.train_label.values
        are_generated = segm_annos.generated.any()

        # OPT II: tensor functions adapted from kuangliu's code
        # https://github.com/kuangliu/torchcv/tree/master/torchcv/transforms

        # convert to torch tensors
        seg_boxes = torch.from_numpy(seg_boxes).float()
        labels = torch.from_numpy(labels)

        if self.use_linemaps:

            if are_generated:
                # incomplete annotations -> use line detections to avoid false negatives
                lbl_im = self.line_map_dict[global_seg_idx]
                # resize to crop
                lbl_im = lbl_im.resize(pil_im.size)
            else:
                # assume all ground truth signs are annotated
                # provide dummy label map
                lbl_im = Image.new('1', pil_im.size, 0)

            if False:
                from skimage.color import label2rgb
                import matplotlib.pyplot as plt

                plt.figure(figsize=(10, 10))
                # plt.imshow(lbl_ind)
                plt.imshow(label2rgb(np.asarray(lbl_im), np.asarray(pil_im)))
                plt.show()

            # crop tile
            # print pil_im.size, seg_boxes.shape, labels.shape, tile_bbox
            im, boxes, labels, linemap = crop_box_lm(pil_im, seg_boxes, labels, lbl_im, tile_bbox)
            # scale tile
            im, boxes, linemap = resize_lm(im, boxes, linemap, None, scale=scale)

            # apply augmentation pipeline and convert from PIL to numpy
            if self.transform is not None:
                im, boxes, labels, linemap = self.transform(im, boxes, labels, linemap)

            return im, boxes, labels, linemap

        else:

            # crop tile
            #print pil_im.size, seg_boxes.shape, labels.shape, tile_bbox
            im, boxes, labels = crop_box(pil_im, seg_boxes, labels, tile_bbox)
            # scale tile
            im, boxes = resize(im, boxes, None, scale=scale)

            # apply augmentation pipeline and convert from PIL to numpy
            if self.transform is not None:
                im, boxes, labels = self.transform(im, boxes, labels)

            return im, boxes, labels

    def __len__(self):
        return len(self.sample2tile_list)


# helper functions

def create_line_map_from_line_anno(line_anno_df_list, coll_idx, segm_idx, input_shape):
    line_height = 3

    # select line annotations
    line_anno_df = line_anno_df_list[coll_idx]
    seg_line_df = line_anno_df[line_anno_df.segm_idx == segm_idx]
    # # collect all line coordinates
    rr, cc, lbboxes = collect_line_coords(seg_line_df, scale=1 / 16.)
    # compute line trafo
    line_trafo = create_line_trafo(rr, cc, input_shape / 16)
    # # compute masks
    line_mask = line_trafo < line_height
    # convert to binary PIL image
    lbl_im = convert2binaryPIL(line_mask)

    return lbl_im


def create_line_map_from_line_det(line_detection_dict, global_seg_idx, scale, input_shape):
    # get line detection
    lbl_ind = line_detection_dict[global_seg_idx]
    # compute line map
    lbl_ind = compute_image_label_map(lbl_ind, np.array(input_shape * scale, dtype=int), padding=5)  # default:16, other padding=16 20 24
    # convert to binary PIL image
    lbl_im = convert2binaryPIL(lbl_ind)

    return lbl_im


# run test
def test(collections=['train'], gen_collections=[], gen_folder=None, use_balanced_idx=True, use_linemaps=False,
         remove_empty_tiles=False, min_align_ratio=0.2, relative_path='../../'):
    ssd_dataset = CuneiformSSD(collections=collections, gen_file_path=None, gen_collections=gen_collections,
                               gen_folder=gen_folder, relative_path=relative_path,
                               use_balanced_idx=use_balanced_idx, tile_shape=[600, 600], use_linemaps=use_linemaps,
                               remove_empty_tiles=remove_empty_tiles, min_align_ratio=min_align_ratio)
    return ssd_dataset
