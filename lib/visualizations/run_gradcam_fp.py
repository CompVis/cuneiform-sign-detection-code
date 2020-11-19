from torchvision import transforms as trafos

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from PIL import Image


from ..visualizations.sign_visuals import visualize_group_detections, crop_detections
from ..models.mobilenetv2_mod03 import MobileNetV2
from ..transliteration.mzl_util import get_unicode_comp


def compute_grad_cam(model_ft, det_crops, det_labels, test_transform, context_pad, device, visualize_grad_cam=False):
    # hook the feature extractor
    features_blobs = []
    gradient_blobs = []

    def hook_feature(module, input, output):
        features_blobs.append(output.detach().cpu())

    def hook_gradient(module, grad_in, grad_out):
        gradient_blobs.append(grad_out[-1].detach().cpu())

    # at the layer of your choice
    my_layer = model_ft.features[-2]
    hook_fwd = my_layer.register_forward_hook(hook_feature)
    hook_bwd = my_layer.register_backward_hook(hook_gradient)

    # run the model
    inputs_blobs = []
    labels_blobs = []

    # set to eval
    model_ft.eval()

    # extract features
    for i, (crop_im, crop_lbl) in tqdm(enumerate(zip(det_crops, det_labels)), total=len(det_crops)):
        model_ft.zero_grad()

        # prepare input
        inputs = test_transform(crop_im).unsqueeze(0)
        labels = torch.tensor(crop_lbl).unsqueeze(0)

        # append
        inputs_blobs.append(inputs.clone())
        labels_blobs.append(crop_lbl)

        inputs = inputs.to(device)
        labels = labels.to(device)

        # compute predictions using the model
        outputs = model_ft(inputs)
        _, preds = torch.max(outputs, 1)

        # use score as gradient
        gradients = torch.gather(outputs, 1, labels.view(-1, 1))
        gradients.sum().backward()

    # remove hooks
    hook_fwd.remove()
    hook_bwd.remove()

    # container for final heatmaps
    heatmap_blobs = []

    # compute grad-cam
    for idx_in_batch in range(len(det_crops)):  # len(inputs_blobs)
        # get pooled gradients and activations
        pooled_gradient = torch.mean(gradient_blobs[idx_in_batch][0], dim=[1, 2])
        activations = features_blobs[idx_in_batch][0]
        # get input image
        # input_im = re_transform(inputs_blobs[idx_in_batch].clone().squeeze(0)).convert('RGB')
        input_im = det_crops[idx_in_batch].copy()

        # get box
        imw, imh = input_im.size[::-1]
        ###bbox = compute_rel_bbox(det_boxes[idx_in_batch], det_crops[idx_in_batch].width)
        bbox = [context_pad, context_pad, imh - context_pad, imw - context_pad]

        # weight channels by corresponding gradients
        cam = torch.mean(pooled_gradient.view(512, 1, 1) * activations, dim=0)
        # torch.mv(activations.reshape(512, -1).transpose(1,0), pooled_gradient).reshape(7,7)
        # relu ontop of heatmap
        if 0:
            cam = torch.relu(cam)
        else:
            cam = torch.abs(cam)
        # resize to input dimensions
        cam = torch.nn.functional.interpolate(cam.unsqueeze(0).unsqueeze(0).float(), size=(imw, imh),
                                              # scale_factor=32,
                                              mode='bilinear', align_corners=False)
        if 0:
            # compute completness
            compl, norm_compl = compute_completness(cam, bbox)
            # append
            list_compl.append(compl)
            list_norm_compl.append(norm_compl)
        else:
            compl = 0
            norm_compl = 0

        # construct heatmap
        # normalize heatmap
        cam /= torch.max(cam)
        # get image shape
        cam_img = cam.numpy().squeeze()
        # store in list
        heatmap_blobs.append(cam_img)

        # visualize
        if visualize_grad_cam:
            print(idx_in_batch)

            # save it as colorimage
            colormap = plt.cm.hot
            heatmap = colormap(cam_img)
            heatmap = Image.fromarray(np.uint8(heatmap * 255))  # 225
            # make the heatmap transparent
            heatmap.putalpha(150)
            # now fuse image and heatmap together
            input_im.paste(heatmap, (0, 0), heatmap)

            # plot heatmap
            fig, ax = plt.subplots()
            ax.imshow(np.asarray(input_im))
            ax.set_title(
                'class: {}  compl: {:.2f} | {:.2f}'.format(labels_blobs[idx_in_batch], compl, norm_compl))
            # plot box
            ax.add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                       fill=False, edgecolor='blue', linestyle='-', alpha=0.3, linewidth=2.0))
            ax.set_axis_off()
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
            plt.show()

    return heatmap_blobs


def visualize_TPFP_heatmap(subgroup_eval_df, det_crops, det_heatmaps, sign_cls, lbl2lbl, cunei_mzl_df, font_prop,
                           context_pad, store_figures=False, tpfp_figure_path=None, fig_desc='FP_heat_pred'):
    subgroup_heatmap = []
    for det_im, det_hm in zip(det_crops, det_heatmaps):
        # create copy (do not alter crop images)
        input_im = det_im.copy()

        # save it as colorimage
        colormap = plt.cm.hot
        heatmap = colormap(det_hm)
        heatmap = Image.fromarray(np.uint8(heatmap * 255))  # 225
        # make the heatmap transparent (lower is more transparent)
        heatmap.putalpha(150)  # 125
        # now fuse image and heatmap together
        input_im.paste(heatmap, (0, 0), heatmap)

        # convert to list of ndarray and append to list
        subgroup_heatmap.append(np.asarray(input_im))

    # call visualize function
    visualize_group_detections(subgroup_eval_df, subgroup_heatmap, lbl2lbl, cunei_mzl_df, font_prop,
                               figs_sz=None, num_cols=1, context_pad=context_pad, max_vis=8)  # num_cols=6
    # create nice plot
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=2.0)  # prevents text from being cut off
    plt.gca().set_axis_off()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if store_figures:
        plt.savefig(tpfp_figure_path.format(sign_cls, fig_desc), bbox_inches='tight', pad_inches=0, dpi=150)
    plt.show()


def gen_fp_gradcam_visuals(col_eval_df, list_sign_detections, dataset, customdidx2didx, relative_path,
                           classifier_model_version, classes_to_show, lbl2lbl, cunei_mzl_df, font_prop,
                           tpfp_figure_path):

    crop_h, crop_w = [224, 224]
    num_c = 1
    num_classes = 240
    gray_mean = [0.5]
    gray_std = [1.0]

    # Data augmentation and normalization for training
    # Just normalization for validation
    test_transform = trafos.Compose([
        trafos.Lambda(lambda x: x.convert('L')),  # comment, if num_c = 3,
        trafos.Resize((crop_h, crop_w), interpolation=Image.BILINEAR),
        trafos.ToTensor(),
        trafos.Normalize(mean=gray_mean * num_c, std=gray_std * num_c),
    ])

    ############################
    # Load model
    arch_opt = 1
    width_mult = 0.625

    # use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load model definition
    model_ft = MobileNetV2(input_size=224, width_mult=width_mult, n_class=num_classes, input_dim=1,
                           arch_opt=arch_opt)
    # load model weights
    weights_path = '{}results/weights/cuneiNet_basic_{}.pth'.format(relative_path, classifier_model_version)
    model_ft.load_state_dict(torch.load(weights_path), strict=False)

    # deploy on device
    model_ft = model_ft.to(device)

    print('num_params:', sum([param.nelement() for param in model_ft.parameters()]))

    ############################
    # Iterate classes to show

    # [30, 57, 113, 80, 108, 13, 82, 100]
    #selected_sign_cls = 30  # 108, 30

    for selected_sign_cls in classes_to_show:
        print(u'sign class {} {}'.format(selected_sign_cls, get_unicode_comp(lbl2lbl[selected_sign_cls], cunei_mzl_df)))

        ############################
        # Data selection

        # select by detection type
        select_tp = (col_eval_df.det_type == 3) & (col_eval_df.max_det)
        select_fp = ~select_tp  # & col_eval_df.pred.isin(eval_fast.list_gt_boxes_df[-1].cls.unique())
        # select by class
        select_classes = col_eval_df.pred.isin(classes_to_show)

        if 0:
            # TP
            # select specific classes and only tp detections
            group_eval_df = col_eval_df[
                select_classes & select_tp].copy()  # explicit copy to add a column on slice without warning
            group_eval_df['group_idx'] = np.arange(0, len(group_eval_df))
            print("valid crops found: {}".format(len(group_eval_df)))

            # select TP detections of class
            subgroup_eval_df = group_eval_df[group_eval_df.true == selected_sign_cls].sort_values('score', ascending=False)
        else:
            # FP
            # select specific classes and only tp detections
            group_eval_df = col_eval_df[
                select_classes & select_fp].copy()  # explicit copy to add a column on slice without warning
            group_eval_df['group_idx'] = np.arange(0, len(group_eval_df))
            print("valid crops found: {}".format(len(group_eval_df)))

            # select FP detections of class
            subgroup_eval_df = group_eval_df[group_eval_df.pred == selected_sign_cls].sort_values('score', ascending=False)

        context_pad = 30  # 10

        group_crops, group_bboxes = crop_detections(group_eval_df, list_sign_detections, dataset, customdidx2didx,
                                                    context_pad=context_pad, return_bboxes=True)
        subgroup_crops = [group_crops[i] for i in subgroup_eval_df.group_idx]
        subgroup_bboxes = [group_bboxes[i] for i in subgroup_eval_df.group_idx]
        print(len(subgroup_crops), subgroup_crops[0].shape)


        ############################
        # Show support for predicted class

        # select if to use ground truth for backprop or predicted label
        # either highlight support of predicted or gt class
        if 1:
            det_labels = subgroup_eval_df.pred.values.astype(long)
        else:
            det_labels = subgroup_eval_df.true.values.astype(long)
            # fall back to pred, if outside of class
            det_labels[det_labels > num_classes] = subgroup_eval_df.pred.values.astype(long)[det_labels > num_classes]

        det_crops = [Image.fromarray(arr) for arr in subgroup_crops]

        ### Run gradCAM
        det_heatmaps = compute_grad_cam(model_ft, det_crops, det_labels, test_transform,
                                        context_pad, device, visualize_grad_cam=False)

        ### Visualize
        if 1:
            print('support for predicted class')
            visualize_TPFP_heatmap(subgroup_eval_df, det_crops, det_heatmaps, selected_sign_cls, lbl2lbl, cunei_mzl_df, font_prop,
                                   context_pad, store_figures=True, tpfp_figure_path=tpfp_figure_path,
                                   fig_desc='FP_heat_pred')


        ############################
        # Show support for GT class

        # either highlight support of predicted or gt class
        if 0:
            det_labels = subgroup_eval_df.pred.values.astype(long)
        else:
            det_labels = subgroup_eval_df.true.values.astype(long)
            # fall back to pred, if outside of class
            det_labels[det_labels > num_classes] = subgroup_eval_df.pred.values.astype(long)[det_labels > num_classes]

        det_crops = [Image.fromarray(arr) for arr in subgroup_crops]

        ### Run gradCAM
        det_heatmaps1 = compute_grad_cam(model_ft, det_crops, det_labels, test_transform,
                                         context_pad, device,  visualize_grad_cam=False)

        ### Visualize
        if 1:
            print('support for GT class')
            visualize_TPFP_heatmap(subgroup_eval_df, det_crops, det_heatmaps1, selected_sign_cls, lbl2lbl, cunei_mzl_df, font_prop,
                                   context_pad, store_figures=True, tpfp_figure_path=tpfp_figure_path,
                                   fig_desc='FP_heat_GT')

        ############################
        # Show difference in support for pred and GT class

        # abs(support(Pred) - support(GT)) (highlights all differences)
        if 0:
            # diff_heatmap = [np.clip(hm2-hm1, 0, 1) for hm1, hm2 in zip(det_heatmaps, det_heatmaps1)]
            diff_heatmap = [np.abs(pred - gt) for pred, gt in zip(det_heatmaps, det_heatmaps1)]
            print('abs(support(Pred) - support(GT))')
            visualize_TPFP_heatmap(subgroup_eval_df, det_crops, diff_heatmap, selected_sign_cls, lbl2lbl, cunei_mzl_df, font_prop,
                                   context_pad, store_figures=False, tpfp_figure_path=tpfp_figure_path,
                                   fig_desc='FP_heat_GT')

        # support(Pred) - support(GT) (highlights parts that go beyond GT class)
        if 1:
            diff_heatmap = [np.clip(pred - gt, 0, 1) for pred, gt in zip(det_heatmaps, det_heatmaps1)]
            # diff_heatmap = [pred-gt for pred, gt in zip(det_heatmaps, det_heatmaps1)]
            print('support(Pred) - support(GT)')
            visualize_TPFP_heatmap(subgroup_eval_df, det_crops, diff_heatmap, selected_sign_cls, lbl2lbl, cunei_mzl_df, font_prop,
                                   context_pad, store_figures=True, tpfp_figure_path=tpfp_figure_path,
                                   fig_desc='FP_heat_GT_deviation')

        if 1:
            diff_heatmap = [(hm - hm.min()) / (hm - hm.min()).max() for hm in diff_heatmap]
            print('support(Pred) - support(GT) [normalized]')
            visualize_TPFP_heatmap(subgroup_eval_df, det_crops, diff_heatmap, selected_sign_cls, lbl2lbl, cunei_mzl_df, font_prop,
                                   context_pad, store_figures=True, tpfp_figure_path=tpfp_figure_path,
                                   fig_desc='FP_heat_GT_deviation_norm')


        # support(GT) - support(Pred) (highlights parts that go beyond pred class)
        if 0:
            diff_heatmap = [np.clip(gt - pred, 0, 1) for pred, gt in zip(det_heatmaps, det_heatmaps1)]
            print('support(GT) - support(Pred)')
            visualize_TPFP_heatmap(subgroup_eval_df, det_crops, diff_heatmap, selected_sign_cls, lbl2lbl, cunei_mzl_df, font_prop,
                                   context_pad, store_figures=False, tpfp_figure_path=tpfp_figure_path,
                                   fig_desc='FP_heat_GT')


        # support(GT) / support(Pred) (highlights deviation from GT)
        if 0:
            # support(GT) / support(Pred) -> looks similar to support(Pred) - support(GT),
            # because everywhere GT > Pred will be set to 1

            # support(Pred) / support(GT) -> looks similar to support(GT) - support(Pred),
            # because everywhere Pred > GT will be set to 1

            # outliers can dominate the visualization (difficult to normalize)

            diff_heatmap = [gt / (pred) for pred, gt in zip(det_heatmaps, det_heatmaps1)]
            print('support(GT) / support(Pred)')
            visualize_TPFP_heatmap(subgroup_eval_df, det_crops, diff_heatmap, selected_sign_cls, lbl2lbl, cunei_mzl_df, font_prop,
                                   context_pad, store_figures=False, tpfp_figure_path=tpfp_figure_path,
                                   fig_desc='FP_heat_GT')

        # support(GT) * support(Pred) (highlights joint parts)
        if 0:
            diff_heatmap = [pred * gt for pred, gt in zip(det_heatmaps, det_heatmaps1)]
            print('support(GT) * support(Pred)')
            visualize_TPFP_heatmap(subgroup_eval_df, det_crops, diff_heatmap, selected_sign_cls, lbl2lbl, cunei_mzl_df, font_prop,
                                   context_pad, store_figures=False, tpfp_figure_path=tpfp_figure_path,
                                   fig_desc='FP_heat_GT')


