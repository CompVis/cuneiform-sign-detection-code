import numpy as np
import pandas as pd
from PIL import Image
from ast import literal_eval
from tqdm import tqdm

import torch.utils.data as data


from ..detection.sign_detection import crop_segment_from_tablet_im, rescale_segment_single
from ..utils.torchcv.transforms.resize import resize


class CuneiformSegments(data.Dataset):
    # lightweight version of cunei_dataset_segments
    # no annotations processing
    # no preloading

    def __init__(self, transform=None, target_transform=None, collection='train', collections=[],
                 relative_path='../', rescale=1.0, only_assigned=True, preload_segments=False):

        self.rescale = rescale
        self.relative_path = relative_path
        self.collection = collection
        self.preload_segments = preload_segments

        # transforms for data preparation
        self.transform = transform
        self.target_transform = target_transform

        if len(collections) > 0:
            # load segment metadata for multiple collections
            df_list = []
            for collection in collections:
                annotation_file = '{}data/segments/tablet_segments_{}.csv'.format(relative_path, collection)
                tablet_segments_df = pd.read_csv(annotation_file, engine='python', index_col=0)
                df_list.append(tablet_segments_df)
            # concatenate to single df
            tablet_segments_df = pd.concat(df_list, ignore_index=True)
        else:
            # load segment metadata for single collection
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
        if only_assigned:
            self.assigned_segments_df = tablet_segments_df[(tablet_segments_df.assigned == True)]
        else:
            self.assigned_segments_df = tablet_segments_df

        # make available for outside
        self.tablet_segments_df = tablet_segments_df

        self.image_data_list = []
        self.sample2seg_list = []
        self.sidx2didx = []
        self.setup_sample_list()

    def setup_sample_list(self, updated_df=None):
        if updated_df is not None:
            self.assigned_segments_df = updated_df

        ### preload segment images
        # crop segment and convert to gray scale
        # IMPORTANT: preload segment crops (without scaling, because memory)
        image_data_list = []
        if self.preload_segments:
            # iterate over segments
            for seg_idx, seg_rec in tqdm(self.assigned_segments_df.iterrows(), total=len(self.assigned_segments_df)):
                # load segment meta data
                image_name, scale, seg_bbox, path_to_image, view_desc = self.get_segment_meta(seg_rec)
                # prepare input tablet
                pil_im = Image.open(path_to_image)
                tablet_seg, new_bbox = crop_segment_from_tablet_im(pil_im, seg_bbox)
                # store in list
                image_data_list.append(tablet_seg)
        self.image_data_list = image_data_list

        self.sample2seg_list = self.assigned_segments_df.index.values

        # map from seg idx to dataset idx
        self.sidx2didx = dict(zip(self.sample2seg_list, range(len(self.sample2seg_list))))

        # setup finished
        print("Setup {} dataset with {} elements".format(self.collection, len(self)))

    def __getitem__(self, index):

        seg_idx = self.sample2seg_list[index]
        seg_rec = self.assigned_segments_df.loc[seg_idx]

        # load segment meta data
        image_name, scale, seg_bbox, path_to_image, view_desc = self.get_segment_meta(seg_rec)

        # specify target
        target = seg_idx

        # get segment image
        if self.preload_segments:
            tablet_seg = self.image_data_list[index]
        else:
            # prepare input tablet
            pil_im = Image.open(path_to_image)
            tablet_seg, new_bbox = crop_segment_from_tablet_im(pil_im, seg_bbox)

        # scale image
        if 0:
            # scale image
            im = rescale_segment_single(tablet_seg, scale)
        else:
            # convert to gray scale
            # tablet_seg = tablet_seg.convert('L')
            # scale segment
            im, _ = resize(tablet_seg, None, None, scale=scale)

        # apply augmentation pipeline and convert from PIL to numpy
        if self.transform is not None:
            im = self.transform(im)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return im, target

    def __len__(self):
        # return total lines
        return len(self.assigned_segments_df)


    def get_segment_meta(self, segment_rec):
        image_name = segment_rec.tablet_CDLI

        # this should control which scale is used in consecutive processing
        scale = segment_rec.scale * self.rescale

        seg_bbox = segment_rec.bbox
        path_to_image = segment_rec.im_path
        view_desc = "{}".format(segment_rec.view_desc).replace("nan", "")

        return image_name, scale, seg_bbox, path_to_image, view_desc

