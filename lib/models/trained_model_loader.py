import torch
import os

from .linenet import LineNet, LineNetFCN

from .mobilenetv2_mod03 import MobileNetV2
from .mobilenetv2_fpn import MobileNetV2FPN

from ..utils.torchcv.models.net import FPNSSD
from ..utils.torchcv.models.rpn_net import RPN


def get_cunei_net_basic(model_version, device, arch_type, arch_opt=1, width_mult=0.5,
                       relative_path='../../', num_classes=240, num_c=1):

    # create classifier model
    basic_net = MobileNetV2(input_size=224, width_mult=width_mult, n_class=num_classes, input_dim=num_c,
                                arch_opt=arch_opt)

    # load pretrained weights
    weights_path = '{}results/weights/cuneiNet_basic_{}.pth'.format(relative_path, model_version)
    basic_net.load_state_dict(torch.load(weights_path))  # , strict=False

    # deploy to device and switch to train
    basic_net.to(device)
    basic_net.eval()  # ATTENTION!

    return basic_net


def get_line_net_fcn(model_version, device, relative_path='../../', num_classes=2, num_c=1):

    # choose model filename
    weights_path = '{}results/weights/lineNet_basic_{}.pth'.format(relative_path, model_version)
    assert os.path.exists(weights_path), "File '{}' not found!".format(weights_path)

    # load model definition
    model_ft = LineNet(num_classes=num_classes, input_channels=num_c)

    # load model weights
    model_ft.load_state_dict(torch.load(weights_path), strict=False)

    # create fully-convolutional version (convolutionalize)
    model_fcn = LineNetFCN(model_ft, num_classes)

    # deploy model to device
    model_fcn = model_fcn.to(device)

    # switch model to evaluation mode
    # model_fcn.train(False)
    model_fcn.eval()

    return model_fcn


def get_fpn_ssd_net(model_version, device, arch_type, with_64, arch_opt=1, width_mult=0.5,
                    relative_path='../../', num_classes=240, num_c=1, rnd_init_model=False):
    # create classifier model
    basic_net = MobileNetV2(input_size=224, width_mult=width_mult, n_class=num_classes, input_dim=num_c,
                            arch_opt=arch_opt)

    # create FPN model with classifier model
    fpn_net = MobileNetV2FPN(basic_net, num_classes=num_classes, width_mult=width_mult, with_p4=with_64)

    # load full detector net
    fpnssd_net = FPNSSD(fpn_net, num_classes)
    if not rnd_init_model:
        # load pretrained weights
        weights_path = '{}results/weights/fpn_net_{}.pth'.format(relative_path, model_version)
        fpnssd_net.load_state_dict(torch.load(weights_path, map_location=device))  # , strict=False

    # deploy to device and switch to train
    fpnssd_net.to(device)
    fpnssd_net.eval()

    return fpnssd_net


def get_rpn_net(model_version, device, arch_type, with_64, arch_opt=1, width_mult=0.5,
                    relative_path='../../', num_classes=240, num_c=1):
    # create classifier model
    basic_net = MobileNetV2(input_size=224, width_mult=width_mult, n_class=num_classes, input_dim=num_c,
                            arch_opt=arch_opt)

    # create FPN model with classifier model
    fpn_net = MobileNetV2FPN(basic_net, num_classes=num_classes, width_mult=width_mult, with_p4=with_64)

    # load full detector net
    rpn_net = RPN(fpn_net, num_classes, with_64)
    # load pretrained weights
    weights_path = '{}results/weights/fpn_net_{}.pth'.format(relative_path, model_version)
    rpn_net.load_state_dict(torch.load(weights_path))  # , strict=False

    # deploy to device and switch to train
    rpn_net.to(device)
    rpn_net.eval()

    return rpn_net

