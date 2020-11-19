'''Encode object boxes and labels.'''
import math
import torch
import numpy as np

from .meshgrid import meshgrid
from .box import box_iou, box_nms, change_box_order


class RetinaBoxCoder:
    def __init__(self, input_size=[512., 512.], with_64=False, create_bg_class=True, with_4_aspects=False, with_4_scales=False):
        self.num_anchors = 12
        # self.anchor_areas = (32*32., 64*64., 128*128., 256*256., 512*512.)  # p3 -> p7
        # self.aspect_ratios = (1/2., 1/1., 2/1.)
        # self.scale_ratios = (1., pow(2,1/3.), pow(2,2/3.))
        self.with_64 = with_64
        if self.with_64:
            self.anchor_areas = [64 * 64., 128 * 128., 256 * 256.]
        else:
            self.anchor_areas = [128 * 128., 256 * 256.]
        if with_4_aspects:
            self.aspect_ratios = [3 / 5., 1 / 1., 2 / 1., 3 / 1.]
        else:
            self.aspect_ratios = [2 / 1., 1 / 1., 2 / 1., 3 / 1.]  # [1 / 0.5, 1 / 1., 2 / 1., 3 / 1.]
        if with_4_scales:
            assert with_4_scales != with_4_aspects, "Cannot use with_4_scales and with_4_aspects simultaneously!"
            self.scale_ratios = [0.8, 1., pow(2, 1 / 3.), pow(2, 2 / 3.)]
            self.aspect_ratios = [1 / 1., 2 / 1., 3 / 1.]
        else:
            self.scale_ratios = [1., pow(2, 1 / 3.), pow(2, 2 / 3.)]

        self.input_size = torch.tensor(input_size).float()
        self.anchor_boxes = self._get_anchor_boxes(input_size=self.input_size)

        self.create_bg_class = create_bg_class

    def _get_anchor_wh(self):
        '''Compute anchor width and height for each feature map.

        Returns:
          anchor_wh: (tensor) anchor wh, sized [#fm, #anchors_per_cell, 2].
        '''
        anchor_wh = []
        for s in self.anchor_areas:
            for ar in self.aspect_ratios:  # w/h = ar
                h = math.sqrt(s / ar)
                w = ar * h
                for sr in self.scale_ratios:  # scale
                    anchor_h = h * sr
                    anchor_w = w * sr
                    anchor_wh.append([anchor_w, anchor_h])
        num_fms = len(self.anchor_areas)
        return torch.Tensor(anchor_wh).view(num_fms, -1, 2)

    def _get_anchor_boxes(self, input_size):
        '''Compute anchor boxes for each feature map.

        Args:
          input_size: (tensor) model input size of (w,h).

        Returns:
          boxes: (list) anchor boxes for each feature map. Each of size [#anchors,4],
                        where #anchors = fmw * fmh * #anchors_per_cell
        '''
        num_fms = len(self.anchor_areas)
        anchor_wh = self._get_anchor_wh()
        # fm_sizes = [(input_size / pow(2., i + 3)).ceil() for i in range(num_fms)]  # p3 -> p7 feature map sizes
        if self.with_64:   # num_fms == 3:
            fm_sizes = [(input_size / pow(2., i + 4)).ceil() for i in range(num_fms)]  # p4 -> p6 feature map sizes
        else:  # num_fms == 2:
            fm_sizes = [(input_size / pow(2., i + 5)).ceil() for i in range(num_fms)]  # p5 -> p6 feature map sizes

        boxes = []
        for i in range(num_fms):
            fm_size = fm_sizes[i]
            grid_size = input_size / fm_size
            fm_w, fm_h = int(fm_size[0]), int(fm_size[1])
            xy = meshgrid(fm_w, fm_h) + 0.5  # [fm_h*fm_w, 2]
            xy = (xy * grid_size).view(fm_h, fm_w, 1, 2).expand(fm_h, fm_w, self.num_anchors, 2)
            wh = anchor_wh[i].view(1, 1, self.num_anchors, 2).expand(fm_h, fm_w, self.num_anchors, 2)
            box = torch.cat([xy - wh / 2., xy + wh / 2.], 3)  # [x,y,x,y]
            boxes.append(box.view(-1, 4))
        return torch.cat(boxes, 0)

    def encode(self, boxes, labels, linemap):
        '''Encode target bounding boxes and class labels.

        We obey the Faster RCNN box coder:
          tx = (x - anchor_x) / anchor_w
          ty = (y - anchor_y) / anchor_h
          tw = log(w / anchor_w)
          th = log(h / anchor_h)

        Args:
          boxes: (tensor) bounding boxes of (xmin,ymin,xmax,ymax), sized [#obj, 4].
          labels: (tensor) object class labels, sized [#obj,].

        Returns:
          loc_targets: (tensor) encoded bounding boxes, sized [#anchors,4].
          cls_targets: (tensor) encoded class labels, sized [#anchors,].
        '''
        anchor_boxes = self.anchor_boxes
        ious = box_iou(anchor_boxes, boxes)
        max_ious, max_ids = ious.max(1)
        boxes = boxes[max_ids]

        # need to check if anchor_box center has positive linemap
        anchor_ctrs = torch.zeros((anchor_boxes.shape[0], 2)).int()
        anchor_ctrs[:, 0] = (anchor_boxes[:, 2] + anchor_boxes[:, 0]) / 2
        anchor_ctrs[:, 1] = (anchor_boxes[:, 3] + anchor_boxes[:, 1]) / 2
        linemap_val = np.asarray(linemap)[anchor_ctrs[:, 1], anchor_ctrs[:, 0]]

        boxes = change_box_order(boxes, 'xyxy2xywh')
        anchor_boxes = change_box_order(anchor_boxes, 'xyxy2xywh')

        loc_xy = (boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:]
        loc_wh = torch.log(boxes[:, 2:] / anchor_boxes[:, 2:])
        loc_targets = torch.cat([loc_xy, loc_wh], 1)
        if self.create_bg_class:
            cls_targets = 1 + labels[max_ids]
        else:
            # if background class 0 already exists in labels
            cls_targets = labels[max_ids]

        cls_targets[max_ious < 0.5] = 0  # WATCH OUT HERE, this is just for testing!!
        # ignore = (max_ious > 0.4) & (max_ious < 0.5)  # ignore ious between [0.4,0.5]
        # cls_targets[ignore] = -1  # mark ignored to -1

        # ignore if box centered on line detection and iou below 0.5
        ignore = torch.from_numpy(linemap_val.astype(np.uint8)) & (max_ious < 0.35)  # 0.5
        cls_targets[ignore] = -1  # mark ignored to -1
        return loc_targets, cls_targets

    def decode(self, loc_preds, cls_preds, input_size, score_thresh=0.5, nms_thresh=0.5):
        '''Decode outputs back to bouding box locations and class labels.

        Args:
          loc_preds: (tensor) predicted locations, sized [#anchors, 4].
          cls_preds: (tensor) predicted class labels, sized [#anchors, #classes].
          input_size: (tuple) model input size of (w,h).

        Returns:
          boxes: (tensor) decode box locations, sized [#obj,4].
          labels: (tensor) class labels for each box, sized [#obj,].
        '''
        CLS_THRESH = score_thresh
        NMS_THRESH = nms_thresh

        input_size = torch.Tensor(input_size)
        # anchor_boxes = self._get_anchor_boxes(input_size)  # xywh
        anchor_boxes = change_box_order(self._get_anchor_boxes(input_size), 'xyxy2xywh')

        loc_xy = loc_preds[:, :2]
        loc_wh = loc_preds[:, 2:]

        xy = loc_xy * anchor_boxes[:, 2:] + anchor_boxes[:, :2]
        wh = loc_wh.exp() * anchor_boxes[:, 2:]
        boxes = torch.cat([xy - wh / 2, xy + wh / 2], 1)  # [#anchors,4]

        score, labels = cls_preds.sigmoid().max(1)  # [#anchors,]
        ids = score > CLS_THRESH
        ids = ids.nonzero().squeeze()  # [#obj,]
        keep = box_nms(boxes[ids], score[ids], threshold=NMS_THRESH)
        return boxes[ids][keep], labels[ids][keep]  # , score[ids][keep]

    def decode_boxes(self, loc_preds):

        anchor_boxes = change_box_order(self.anchor_boxes, 'xyxy2xywh')

        loc_xy = loc_preds[:, :2]
        loc_wh = loc_preds[:, 2:]

        xy = loc_xy * anchor_boxes[:, 2:] + anchor_boxes[:, :2]
        wh = loc_wh.exp() * anchor_boxes[:, 2:]
        box_preds = torch.cat([xy - wh / 2, xy + wh / 2], 1)

        boxes = box_preds
        return boxes
