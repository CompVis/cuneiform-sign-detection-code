import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


# HELPER FUNCTIONS

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # m.weight.data.normal_(0, math.sqrt(2. / n))
            # init.xavier_normal(m.weight.data)
            # init.kaiming_normal(m.weight.data)
            init.normal_(m.weight.data, std=0.01)
            # check if bias = True
            if hasattr(m.bias, 'data'):
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.005)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            # check if affine = True
            if hasattr(m.bias, 'data'):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def copy_layer_params(target, source):
    """ Copy layer parameters from source to target; size of arrays needs to match! """
    target.weight.data.copy_(source.weight.data.view(target.weight.size()))
    target.bias.data.copy_(source.bias.data.view(target.bias.size()))


# HELPER MODULES


class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=False):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if self.ACROSS_CHANNELS:
            # make it work with pytorch 0.2.X # hacky!!! should be ConstantPadding
            # self.average = nn.Sequential(
            #     nn.ReplicationPad3d(padding=(0, 0, 0, 0, int((local_size - 1.0) / 2), int((local_size - 1.0) / 2))),
            #     nn.AvgPool3d(kernel_size=(local_size, 1, 1), stride=1),
            # )
            self.average = nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                                        stride=1,
                                        padding=(int((local_size - 1.0) / 2), 0, 0))
        else:
            self.average = nn.AvgPool2d(kernel_size=local_size,
                                        stride=1,
                                        padding=int((local_size - 1.0) / 2))
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x


class Softmax3D(nn.Module):
    def forward(self, input_):
        batch_size = input_.size()[0]
        output_ = torch.stack([F.softmax(input_[i]) for i in range(batch_size)], 0)
        return output_


# MAIN MODULES


class LineNet(nn.Module):

    def __init__(self, num_classes=1000, input_channels=3):
        super(LineNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            LRN(alpha=1e-4, beta=0.75, local_size=1),
            nn.Conv2d(64, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            LRN(alpha=1e-4, beta=0.75, local_size=1),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384, affine=False, momentum=.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384, affine=False, momentum=.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, affine=False, momentum=.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.fc6 = nn.Linear(256 * 6 * 6, 512)
        self.score = nn.Linear(512, 240)
        self.classifier = nn.Sequential(
            self.fc6,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            self.score,
        )
        self.line_score = nn.Linear(512, num_classes)
        self.line_classifier = nn.Sequential(
            self.fc6,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            self.line_score,
        )
        initialize_weights(self)

    def legacy_forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)  # x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)  # x = x.view(x.size(0), -1)
        x = self.line_classifier(x)

        return x


class LineNetFCN(nn.Module):
    def __init__(self, original_model, num_classes=240):
        super(LineNetFCN, self).__init__()

        # simple assign !no copy! (could use copy.deepcopy(), but assume original_model is not used anymore)
        self.features = original_model.features

        # create new module and assign features
        # original_net = CuneiNet(input_channels=1)
        # self.features = original_net.features
        # self.features.load_state_dict(original_model.features.state_dict())

        # softmax function
        self.softmax = nn.Softmax2d()    ## Softmax3D(),

        # create fcn head
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=6, padding=0),
            nn.ReLU(inplace=True),
            # nn.Dropout(), DO NOT USE 1d dropout!!!
            nn.Dropout2d(),
            nn.Conv2d(512, num_classes, kernel_size=1, padding=0),
            # self.softmax  # not here to
        )

        # perform net surgery
        self.net_surgery(original_model)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = self.softmax(x)
        # batch_size = x.size()[0]
        # x = torch.stack([F.softmax(x[i]) for i in range(batch_size)], 0)
        return x

    def get_conv_features(self, x):
        x = self.features(x)
        return x

    def get_fc_features(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def net_surgery(self, original_model):
        """ perform net surgery
                original.classifier --> fcn.classifier
        """
        for i, l1 in enumerate(original_model.line_classifier):
            if isinstance(l1, nn.Linear):
                l2 = self.classifier[i]
                # l2.weight.data.copy_(l1.weight.data.view(l2.weight.size()))
                # l2.bias.data.copy_(l1.bias.data.view(l2.bias.size()))
                copy_layer_params(l2, l1)
