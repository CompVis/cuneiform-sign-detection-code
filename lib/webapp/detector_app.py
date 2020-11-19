# This script loads a pre-trained cuneiform sign detector and handles detection request from the web demo interface.
# It utilizes Flask in order to make the detector available as a webapp at a specific route (URL).
# The detector used by the web demo is made available under
# http://localhost:PORT/detector_php.


# flask tutorials:
# https://pythonise.com/feed/flask/working-with-json-in-flask
# https://www.tutorialspoint.com/flask/flask_quick_guide.htm

from flask import Flask, render_template, request, make_response, jsonify
from werkzeug.utils import secure_filename

import os
import sys
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

import matplotlib.pyplot as plt


#####################
# global config

sign_model_version = 'vF'   # weakly supervised: vA, semi-supervised: vF
relative_path = '../../'

# network config
arch_type = 'mobile'  # resnet, mobile
arch_opt = 1
width_mult = 0.625  # 0.5 0.625 0.75

with_64 = False
with_4_aspects = False
create_bg_class = False

num_classes = 240

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'


test_nms_thresh = 0.5
test_min_score_thresh = 0.1


# placeholder
sign_model = None
lbl2lbl = None


#####################
# start web app
app = Flask(__name__)


#####################
# initialization

def load_detector():
    #load Model
    fpnssd_net = get_fpn_ssd_net(sign_model_version, device, arch_type, with_64, arch_opt, width_mult,
                                 relative_path, num_classes, num_c=1)
    return fpnssd_net


@app.before_first_request
def setup_app():
    global sign_model
    global lbl2lbl
    # init stuff for app
    print('init my stuff')
    # load detector
    sign_model = load_detector()
    # load labels dict
    lbl2lbl = get_lbl2lbl(relative_path + 'data/newLabels.json')


# # alternative to before_first_request  (one way to execute something after app.run)
# def setup_app(app):
#     pass
# # run setup
# setup_app(app)


#####################
# detector routing

def get_detections(fpnssd_net, device, seg_im, with_64, with_4_aspects,
                   create_bg_class, test_nms_thresh, test_min_score_thresh):
    # prepare box coder
    # box_coder = RetinaBoxCoder()
    box_coder = FPNSSDBoxCoder(input_size=seg_im.size, with_64=with_64, with_4_aspects=with_4_aspects,
                               create_bg_class=create_bg_class)

    # prepare input
    inputs = transforms.Compose([transforms.Lambda(lambda x: x.convert('L')),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.5], std=[1.0])])(seg_im)
    inputs = inputs.unsqueeze(0)

    with torch.no_grad():
        loc_preds, cls_preds = fpnssd_net(inputs.to(device))

        box_preds, label_preds, score_preds = box_coder.decode(
            loc_preds.cpu().data.squeeze(),
            F.softmax(cls_preds.squeeze(), dim=1).cpu().data,
            score_thresh=test_min_score_thresh, nms_thresh=test_nms_thresh)

    # convert detections to all boxes format
    all_boxes = prepare_ssd_outputs_for_eval(box_preds, label_preds, score_preds)

    return all_boxes, box_preds, label_preds, score_preds


def scale_image(pil_im, scale=1.0):
    # scale segment
    w, h = pil_im.size
    ow = int(w * scale)
    oh = int(h * scale)
    return transforms.functional.resize(pil_im, (oh, ow), Image.BILINEAR)


# @app.route('/detect')
# def detect_html():
#     return render_template('sign_detect.html')


# @app.route('/detector', methods=['GET', 'POST'])
# def detect_file():
#     global sign_model
#     global lbl2lbl
#     upload_success = False
#     sec_file_name = None
#     if request.method == 'POST':
#         f = request.files['myFile']
#         # check if there is a file at all
#         if not f.filename == "":
#             # store file
#             sec_file_name = secure_filename(f.filename)
#             f.save(sec_file_name)
#             tab_scale = float(request.form['tab_scale'])
#
#             # load composite image
#             try:
#                 pil_im = Image.open(sec_file_name)
#             except IOError:
#                 print('could not read image: {}'.format(sec_file_name))
#
#
#             # ensure that target size is in bounds
#             max_num_pixels = 81e6  # crashes with 6GB around 82e6
#             min_edge_length = 224
#             imw, imh = pil_im.size
#             trgtw = (tab_scale * imw)
#             trgth = (tab_scale * imh)
#
#             print("resolution: {} x {} in bounds: {} [{} vs {}]".format(trgtw, trgth, (trgth * trgtw) < max_num_pixels,
#                                                                         trgth * trgtw, max_num_pixels))
#
#             if trgtw * trgth < max_num_pixels and (trgtw >= min_edge_length and trgth >= min_edge_length):
#
#                 upload_success = True
#
#                 # scale segment
#                 pil_im = scale_image(pil_im, scale=tab_scale)
#
#                 # run detector
#                 (all_boxes, box_preds,
#                  label_preds, score_preds) = get_detections(sign_model, device, pil_im, with_64, with_4_aspects,
#                                                             create_bg_class, test_nms_thresh, test_min_score_thresh)
#
#                 if 1:
#                     # for plots
#                     input_im = np.asarray(pil_im)
#
#                     # plot prediction
#                     plt.figure(figsize=(10, 10))
#                     plot_boxes(box_preds, confidence=score_preds)
#                     plt.imshow(input_im, cmap='gray')
#                     plt.grid(True, color='w', linestyle=':')
#                     plt.gca().set_axis_off()
#                     plt.gca().xaxis.set_major_locator(plt.NullLocator())
#                     plt.gca().yaxis.set_major_locator(plt.NullLocator())
#                     plt.savefig('./static/detection_res.png', bbox_inches='tight', pad_inches=0, dpi=75)
#
#                 # web_export
#                 if 1:
#                     res_name = sec_file_name.split('.')[0]  # "{}{}".format(image_name, view_desc)
#                     saa_version = 'dummy'
#                     res_export = "results_web_export/{}_detections_ssd/{}".format(sign_model_version, saa_version)
#
#                     # reverse shift (due to center crop) and/or scaling (for better detections)
#                     # (important when exporting detections based on original image size)
#                     # (in case of plotting with respect to input image, no useful)
#                     rev_scaling = 1. / tab_scale
#                     for cls_boxes in all_boxes:
#                         cls_boxes = cls_boxes[0]
#                         if len(cls_boxes) > 0:
#                             cls_boxes[:, :4] *= rev_scaling
#
#                     # check folder
#                     make_folder(res_export)
#
#                     # save all_boxes for web export
#                     outfile = "{}/{}_all_boxes.csv".format(res_export, res_name)
#                     export_detections_to_web(outfile, all_boxes, lbl2lbl)
#
#     return render_template('detector_res.html', file_name=sec_file_name, upload_success=upload_success)


@app.route('/detector_php', methods=['GET', 'POST'])
def detect_file_php():
    global sign_model
    global lbl2lbl

    # negative response
    response = {"detection": False}

    if request.method == 'POST':

            tab_scale = float(request.form['tab_scale'])
            det_path = request.form['det_path']
            im_path = request.form['im_path']

            # load composite image
            try:
                pil_im = Image.open(im_path)
            except IOError:
                print('could not read image: {}'.format(im_path))

            # ensure that target size is in bounds
            max_num_pixels = 81e6  # crashes with 6GB around 82e6
            min_edge_length = 224
            imw, imh = pil_im.size
            trgtw = (tab_scale * imw)
            trgth = (tab_scale * imh)

            print("resolution: {} x {} in bounds: {} [{} vs {}]".format(trgtw, trgth, (trgth * trgtw) < max_num_pixels,
                                                                        trgth * trgtw, max_num_pixels))

            if trgtw * trgth < max_num_pixels and (trgtw >= min_edge_length and trgth >= min_edge_length):

                # scale segment
                pil_im = scale_image(pil_im, scale=tab_scale)

                # run detector
                (all_boxes, box_preds,
                 label_preds, score_preds) = get_detections(sign_model, device, pil_im, with_64, with_4_aspects,
                                                            create_bg_class, test_nms_thresh, test_min_score_thresh)

                # web_export
                if 1:
                    # reverse shift (due to center crop) and/or scaling (for better detections)
                    # (important when exporting detections based on original image size)
                    # (in case of plotting with respect to input image, no useful)
                    rev_scaling = 1. / tab_scale
                    for cls_boxes in all_boxes:
                        cls_boxes = cls_boxes[0]
                        if len(cls_boxes) > 0:
                            cls_boxes[:, :4] *= rev_scaling

                    # check folder
                    make_folder(det_path.rsplit('/', 1)[0])

                    # save all_boxes for web export
                    export_detections_to_web(det_path, all_boxes, lbl2lbl)

                # positive response
                response = {"detection": True}

    # return make_response(jsonify({"detection": True, "error": None}), 200)
    return response


#####################
# main function

if __name__ == '__main__':

    # ensure that parent path is on the python path in order to have all packages available
    relative_path = '../../'
    parent_path = os.path.join(os.getcwd(), relative_path)
    parent_path = os.path.realpath(parent_path)  # os.path.abspath(...)
    sys.path.insert(0, parent_path)

    # own stuff
    from lib.models.trained_model_loader import get_fpn_ssd_net
    from lib.visualizations.sign_visuals import plot_boxes
    from lib.utils.torchcv.box_coder_fpnssd import FPNSSDBoxCoder
    from lib.evaluations.sign_evaluation_prep import prepare_ssd_outputs_for_eval
    from lib.webapp.web_io import export_detections_to_web, make_folder
    from lib.transliteration.sign_labels import get_lbl2lbl

    # run web app
    # for config see: https://flask.palletsprojects.com/en/1.1.x/api/#flask.Flask.run
    app.run(debug=False, port=5000)  # public: host='0.0.0.0', port=5001


