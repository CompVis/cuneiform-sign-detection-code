import pandas as pd
from future.utils import iteritems
from tqdm import tqdm
from ast import literal_eval

from PIL import Image
import torch.utils.data as data

from ..utils.bbox_utils import *
from ..utils.transform_utils import crop_pil_image, spatial_sample

DEBUG_MODE = False


class CuneiformCollection(data.Dataset):

    def __init__(self, params, transform=None, target_transform=None, relative_path='../', split='train', top_k=-1, top_k_pick=-1, pad_to_square=True):

        self.gray_mean = params['gray_mean']
        self.context_pad = params['context_pad']
        if 'test' in split:
            self.context_pad = 0  # no padding needed
        self.num_classes = params['num_classes']
        self.min_align_ratio = 0.6
        if 'min_align_ratio' in params:
            self.min_align_ratio = params['min_align_ratio']

        # transforms for data preparation
        self.transform = transform
        self.target_transform = target_transform
        self.pad_to_square = pad_to_square

        self.compl_thresh, self.ncompl_thresh = -1, -1
        if 'compl_thresh' in params:
            self.compl_thresh = params['compl_thresh']
        if 'ncompl_thresh' in params:
            self.ncompl_thresh = params['ncompl_thresh']

        # load annotations
        annotation_file = '{}data/annotations/bbox_annotations_{}.csv'.format(relative_path, split)
        meta_df = pd.read_csv(annotation_file, engine='python')  # read annotation file

        # additional annos (investigate impact of additional train data)
        if 'train' in split and 'extra_collections' in params:
            list_annos = [meta_df]
            for collection in params['extra_collections']:
                annotation_file = '{}data/annotations/bbox_annotations_{}.csv'.format(relative_path, collection)
                anno_df = pd.read_csv(annotation_file, engine='python')  # read annotation file
                list_annos.append(anno_df)
            meta_df = pd.concat(list_annos, ignore_index=True)

        # add missing columns to meta_df
        nd_bbox = np.array(meta_df['bbox'].apply(literal_eval).tolist())  # convert to ndarray
        meta_df['x1'] = nd_bbox[:, 0]
        meta_df['y1'] = nd_bbox[:, 1]
        meta_df['x2'] = nd_bbox[:, 2]
        meta_df['y2'] = nd_bbox[:, 3]
        meta_df['imageName'] = meta_df['tablet_CDLI'] + '.jpg'
        meta_df['image_path'] = '{}data/images/'.format(relative_path) + meta_df['collection'] \
                                + '/' + meta_df['imageName']

        ### load and prepare gen_df
        # append with gen alignments
        gen_cols = ['imageName', 'folder', 'image_path', 'label', 'train_label',
                    'x1', 'y1', 'x2', 'y2', 'width', 'height', 'segm_idx',
                    'line_idx', 'pos_idx', 'det_score', 'm_score', 'align_ratio', 'nms_keep', 'compl', 'ncompl']

        # segm_idx,tablet_CDLI,view_desc,collection,mzl_label,train_label,bbox,relative_bbox

        collections_ext = [split]

        if 'train' in split:
            # OPT I : use csv file that contains list of generated boxes
            if 'gen_file' in params:
                gen_df = pd.read_csv(params['gen_file'], engine='python', header=None,  delimiter=', ', names=gen_cols)  # delimiter might need to be removed?!
            # OPT II : load csv files for collection specific collections and concatenate
            elif 'gen_collections' in params:
                assert params['gen_folder'] is not None, 'When using gen_collections, user needs to provide gen_model!'
                df_list = []
                for gen_coll in params['gen_collections']:
                    gen_file_path = "{}results/{}line_generated_bboxes_refined80_{}.csv".format(relative_path,
                                                                                                params['gen_folder'], gen_coll)
                    gen_df = pd.read_csv(gen_file_path, delimiter=',\s*', engine='python', header=None, names=gen_cols)   # delimiter=', ', delimiter=',\s*',
                    df_list.append(gen_df)
                gen_df = pd.concat(df_list, ignore_index=True)
            # prepare gen_df
            if ('gen_file' in params) or ('gen_collections' in params):

                # IMPORTANT: filter gen data according to align ratio
                gen_df = gen_df[gen_df.align_ratio > self.min_align_ratio]

                # IMPORTANT: fill nan values in a way that avoids filtering
                gen_df.compl = gen_df.compl.fillna(50)
                gen_df.ncompl = gen_df.ncompl.fillna(100)

                num_before_filter = len(gen_df)
                if self.compl_thresh > -1:
                    # filter using compl
                    gen_df = gen_df[gen_df.compl > self.compl_thresh]  # 0, 2, 4, 5
                    print('Completeness {} :: Removed {} samples. [{}]'.format(self.compl_thresh,
                                                                               num_before_filter - len(gen_df),
                                                                               len(gen_df)))
                elif self.ncompl_thresh > -1:
                    # filter using compl
                    gen_df = gen_df[gen_df.ncompl > self.ncompl_thresh]  # 0, 2, 4, 5
                    print('Completeness (norm.) {} :: Removed {} samples. [{}]'.format(self.ncompl_thresh,
                                                                                       num_before_filter - len(gen_df),
                                                                                       len(gen_df)))
                print('class sample count stats: ')
                print(gen_df.train_label.value_counts().describe())

                # add/update additional columns
                gen_df['collection'] = gen_df.folder.str.split('/').str[0]
                gen_df['generated'] = True
                gen_df['imageName'] = gen_df['imageName'].astype(str) + '.jpg'

                # identify all collections with generated annotations
                list_gen_collection = gen_df.collection.unique().tolist()
                collections_ext += list_gen_collection

                # concatenate
                meta_df = pd.concat([meta_df, gen_df], ignore_index=True)

        # drop outlier classes for now (dirty fix)
        class_outlier_select = meta_df.train_label < 240
        if np.any(class_outlier_select):
            print('Drop {} outlier samples!'.format(np.sum(~class_outlier_select)))
            meta_df = meta_df[class_outlier_select]

        # reset index
        self.meta_df = meta_df.reset_index(drop=True)

        # make sure there is width and height
        self.meta_df['width'] = self.meta_df['x2'] - self.meta_df['x1'] + 1
        self.meta_df['height'] = self.meta_df['y2'] - self.meta_df['y1'] + 1

        # only keep top 100 classes
        if top_k > 0:
            top_labels = self.meta_df.label.value_counts()[:top_k].index.values
            top_select = self.meta_df.label.isin(top_labels)
            self.meta_df = self.meta_df[top_select].reset_index()
            if top_k > top_k_pick >= 0:
                print(top_labels)
                print('Only select samples from class {}'.format(top_labels[top_k_pick]))
                class_select = self.meta_df.label == top_labels[top_k_pick]
                self.meta_df = self.meta_df[class_select].reset_index(drop=True)

        # all annotations are used
        self.osd_valid_ind = self.meta_df.index

        # crop pre-processing
        # save longest side of each sign
        self.meta_df['square'] = self.meta_df[['width', 'height']].max(axis=1)
        # for each tablet compute median of longest side, and assign it to each sign
        median_table = self.meta_df[self.meta_df.train_label > 0].groupby('imageName')[['square']].median()
        self.meta_df = self.meta_df.join(median_table, on='imageName', rsuffix='_md')
        # self.meta_df['square_new'] = self.meta_df[['square', 'square_md']].max(axis=1)

        # pre-load all images
        self.use_preload = True
        if self.use_preload:
            map = {key: value for (key, value) in enumerate(self.meta_df['image_path'][self.osd_valid_ind].unique())}
            inv_map = {value: key for key, value in iteritems(map)}  # use items
            self.meta_df['mem_idx'] = self.meta_df['image_path'].replace(inv_map)
            self.image_data_list = []
            for key, impath in tqdm(iteritems(map), total=len(map)):
                im_ref = None
                try:
                    im_ref = Image.open(impath)
                except IOError:
                    print('could not read image: {}'.format(impath))
                # due to memory constraints not .convert('RGB')
                im_ref = im_ref.convert('L')
                self.image_data_list.append(im_ref)

        # setup finished
        print("Setup {} dataset spanning {} collections.".format(split, collections_ext))
        num_segs = self.meta_df['image_path'].nunique()
        print("Select {} bboxes from {} tablets.".format(len(self), num_segs))

    def __getitem__(self, index):

        # map index to csv index
        csv_idx = self.osd_valid_ind[index]

        impath = self.meta_df.iloc[csv_idx]['image_path']
        target = self.meta_df.iloc[csv_idx]['train_label']

        square = self.meta_df.iloc[csv_idx]['square']
        square_md = self.meta_df.iloc[csv_idx]['square_md']

        # load image data
        if self.use_preload:
            mem_idx = self.meta_df.iloc[csv_idx]['mem_idx']
            im_ref = self.image_data_list[mem_idx]
        else:
            im_ref = None
            try:
                im_ref = Image.open(impath).convert('L')  # due to memory constraints not .convert('RGB')
            except IOError:
                print('could not read image: {}'.format(impath))

        # bounding box meta
        bb = [self.meta_df.iloc[csv_idx]['x1'], self.meta_df.iloc[csv_idx]['y1'],
              self.meta_df.iloc[csv_idx]['x2'], self.meta_df.iloc[csv_idx]['y2']]

        # context crop
        context_pad = self.context_pad  # int(square * self.context_pad)  #

        if self.pad_to_square:
            # if background, context_pad = 0
            if target == 0:
                context_pad = 0
            # if largest side of bbox is smaller than median of tablet, add additional context pad
            elif square_md > square:
                context_pad += (square_md - square) / 2.  # divide by 2, because w,h of im_pad grow by 2 * context_pad

        # new fast
        im, bb_pad = crop_pil_image(im_ref, bb, context_pad=context_pad, pad_to_square=self.pad_to_square)

        # apply augmentation pipeline and convert from PIL to numpy
        if self.transform is not None:
            im = self.transform(im)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return im, target

    def __len__(self):
        return len(self.osd_valid_ind)


def test(params, split='train', top_k=-1, top_k_pick=-1, pad_to_square=True, relative_path='../../'):
    dataset = CuneiformCollection(params, relative_path=relative_path, split=split, top_k=top_k, top_k_pick=top_k_pick, pad_to_square=pad_to_square)

    return dataset
