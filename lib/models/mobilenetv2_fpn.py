import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MobileNetV2FPN(nn.Module):
    def __init__(self, original_model, num_classes=240, width_mult=1, with_p4=False):
        super(MobileNetV2FPN, self).__init__()

        # simple assign !no copy! (could use copy.deepcopy(), but assume original_model is not used anymore)
        self.features = original_model.features

        self.conv6 = nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1)

        # Top-down layers
        self.toplayer = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)

        self.with_p4 = with_p4
        if self.with_p4:
            # Lateral layers
            self.latlayer1 = nn.Conv2d(int(32*width_mult), 256, kernel_size=1, stride=1, padding=0)

            # Smooth layers
            self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # init weights (exclude features) TODO
        self._initialize_weights(['conv6', 'toplayer', 'latlayer1', 'smooth1'])

    def forward(self, x):
        for i in range(3):
            x = self.features[i](x)
        c4 = self.features[3](x)  # 14 x 32*width_mult (expansion factor does not affect output of block)

        x = self.features[4](c4)
        x = self.features[5](x)
        x = self.features[6](x)  # 7 x 160*width_mult
        c5 = self.features[7](x)  # 7 x 512
        p6 = self.conv6(c5)

        # Top-down
        p5 = self.toplayer(c5)

        if self.with_p4:
            p4 = self._upsample_add(p5, self.latlayer1(c4))
            p4 = self.smooth1(p4)
            return p4, p5, p6
        else:
            return p5, p6

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def _initialize_weights(self, name_list):
        for name, m in self.named_modules():
            # only init modules in name_list
            if name in name_list:
                # exclude self.features, Mobile_blocks
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


