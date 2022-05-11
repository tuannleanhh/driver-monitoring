import torch
import torch.nn as nn

from .common import Conv, Concat, C3


class NeckYoloV5N(nn.Module):
    def __init__(self, width_multiple):
        super(NeckYoloV5N, self).__init__()
        self.conv1 = Conv(int(512 * width_multiple), int(128 * width_multiple), 1, 1)
        self.up1 = nn.Upsample(None, 2, 'nearest')
        self.cat1 = Concat(dimension=1)
        self.c3_1 = C3(int(256 * width_multiple) + int(128 * width_multiple), int(128 * width_multiple), 1, False)

        self.conv2 = Conv(int(128 * width_multiple), int(128 * width_multiple), 1, 1)
        self.up2 = nn.Upsample(None, 2, 'nearest')
        self.cat2 = Concat(dimension=1)
        self.c3_2 = C3(int(128 * width_multiple) + int(128 * width_multiple), int(128 * width_multiple), 1, False)

        self.conv3 = Conv(int(128 * width_multiple), int(128 * width_multiple), 3, 2)
        self.cat3 = Concat(dimension=1)
        self.c3_3 = C3(int(128 * width_multiple) + int(128 * width_multiple), int(128 * width_multiple), 1, False)

        self.conv4 = Conv(int(128 * width_multiple), int(128 * width_multiple), 3, 2)
        self.cat4 = Concat(dimension=1)
        self.c3_4 = C3(int(128 * width_multiple) + int(128 * width_multiple), int(128 * width_multiple), 1, False)

    def forward(self, features):
        x1 = features[:, :64 * 40 * 40].reshape(-1, 64, 40, 40).contiguous()
        x2 = features[:, 64 * 40 * 40:64 * 40 * 40 + 128 * 20 * 20].reshape(-1, 128, 20, 20).contiguous()
        x3 = features[:, 64 * 40 * 40 + 128 * 20 * 20:].reshape(-1, 256, 10, 10).contiguous()
        x_1 = self.conv1(x3)
        x = self.up1(x_1)
        x = self.cat1((x, x2))
        x = self.c3_1(x)

        x_2 = self.conv2(x)
        x = self.up2(x_2)
        x = self.cat2((x, x1))
        y1 = self.c3_2(x)

        x = self.conv3(y1)
        x = self.cat3((x, x_2))
        y2 = self.c3_3(x)

        x = self.conv4(y2)
        x = self.cat4((x, x_1))
        y3 = self.c3_4(x)

        y1 = y1.reshape(-1, 64 * 40 * 40)
        y2 = y2.reshape(-1, 64 * 20 * 20)
        y3 = y3.reshape(-1, 64 * 10 * 10)

        return torch.cat((y1, y2, y3), dim=1)
