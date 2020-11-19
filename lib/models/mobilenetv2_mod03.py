import torch.nn as nn
import math


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileBlock(nn.Module):
    # introduced to simplify creation of FPN
    def __init__(self, residual_setting, input_channel, output_channel):
        super(MobileBlock, self).__init__()
        t, c, n, s = residual_setting

        block_seq = []
        for i in range(n):
            if i == 0:
                block_seq.append(InvertedResidual(input_channel, output_channel, s, t))
            else:
                block_seq.append(InvertedResidual(input_channel, output_channel, 1, t))
            input_channel = output_channel
        self.mobile_block = nn.Sequential(*block_seq)
        self.output_channel = output_channel

    def forward(self, x):
        return self.mobile_block(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, input_dim=1, width_mult=1., arch_opt=1):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 2],
            #[1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 1],
            # [6, 320, 1, 1],
        ]

        # set arch option
        self.arch_opt = arch_opt

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(32 * width_mult)

        # self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        if self.arch_opt == 1:
            self.last_channel = int(512 * width_mult) if width_mult > 1.0 else 512
        elif self.arch_opt == 2:
            self.last_channel = int(256 * width_mult) if width_mult > 1.0 else 256

        self.features = [conv_bn(input_dim, input_channel, 2)]
        # building inverted residual blocks
        for ii, residual_setting in enumerate(self.interverted_residual_setting):
            t, c, n, s = residual_setting
            output_channel = int(c * width_mult)
            new_block = MobileBlock(residual_setting, input_channel, output_channel)
            self.features.append(new_block)
            input_channel = new_block.output_channel

        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))

        if self.arch_opt == 1:
            self.features.append(nn.AvgPool2d(input_size / 32, stride=1))
            # need stride=1 for FCN (because default stride is kernel_sz)
            # self.features.append(nn.MaxPool2d(kernel_size=input_size / 32, stride=1))

            # building classifier
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(self.last_channel, n_class),
            )

        elif self.arch_opt == 2:
            # building classifier
            self.classifier = nn.Sequential(
                nn.Linear(self.last_channel * 7 * 7, 384),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(384, n_class),
            )

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        self._initialize_weights()

    def forward(self, x):
        for layer in self.features:
            x = layer(x)

        # x = x.view(-1, self.last_channel)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


