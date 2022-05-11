import torch
import torch.nn as nn

from .common import StemBlock, ShuffleV2Block


class BackboneYoloV5N(nn.Module):
    def __init__(self, input_channel, width_multiple):
        super(BackboneYoloV5N, self).__init__()
        # 0-P2/4
        self.stem_0 = StemBlock(c1=input_channel, c2=int(32 * width_multiple), k=3, s=2)
        # 1-P3/8
        self.shuffle_block_1_0 = ShuffleV2Block(int(32 * width_multiple), int(128 * width_multiple), 2)
        # 2
        self.shuffle_block_2_0 = ShuffleV2Block(int(128 * width_multiple), int(128 * width_multiple), 1)
        self.shuffle_block_2_1 = ShuffleV2Block(int(128 * width_multiple), int(128 * width_multiple), 1)
        self.shuffle_block_2_2 = ShuffleV2Block(int(128 * width_multiple), int(128 * width_multiple), 1)
        # 3-P4/16
        self.shuffle_block_3 = ShuffleV2Block(int(128 * width_multiple), int(256 * width_multiple), 2)
        # 4
        self.shuffle_block_4_0 = ShuffleV2Block(int(256 * width_multiple), int(256 * width_multiple), 1)
        self.shuffle_block_4_1 = ShuffleV2Block(int(256 * width_multiple), int(256 * width_multiple), 1)
        self.shuffle_block_4_2 = ShuffleV2Block(int(256 * width_multiple), int(256 * width_multiple), 1)
        self.shuffle_block_4_3 = ShuffleV2Block(int(256 * width_multiple), int(256 * width_multiple), 1)
        self.shuffle_block_4_4 = ShuffleV2Block(int(256 * width_multiple), int(256 * width_multiple), 1)
        self.shuffle_block_4_5 = ShuffleV2Block(int(256 * width_multiple), int(256 * width_multiple), 1)
        self.shuffle_block_4_6 = ShuffleV2Block(int(256 * width_multiple), int(256 * width_multiple), 1)
        # 5-P5/32
        self.shuffle_block_5_0 = ShuffleV2Block(int(256 * width_multiple), int(512 * width_multiple), 2)
        # 6
        self.shuffle_block_6_0 = ShuffleV2Block(int(512 * width_multiple), int(512 * width_multiple), 1)
        self.shuffle_block_6_1 = ShuffleV2Block(int(512 * width_multiple), int(512 * width_multiple), 1)
        self.shuffle_block_6_2 = ShuffleV2Block(int(512 * width_multiple), int(512 * width_multiple), 1)

    def forward(self, x):
        x = self.stem_0(x)
        x = self.shuffle_block_1_0(x)
        x = self.shuffle_block_2_0(x)
        x = self.shuffle_block_2_1(x)
        x1 = self.shuffle_block_2_2(x)

        x = self.shuffle_block_3(x1)
        x = self.shuffle_block_4_0(x)
        x = self.shuffle_block_4_1(x)
        x = self.shuffle_block_4_2(x)
        x = self.shuffle_block_4_3(x)
        x = self.shuffle_block_4_4(x)
        x = self.shuffle_block_4_5(x)
        x2 = self.shuffle_block_4_6(x)

        x = self.shuffle_block_5_0(x2)
        x = self.shuffle_block_6_0(x)
        x = self.shuffle_block_6_1(x)
        x3 = self.shuffle_block_6_2(x)

        x1 = x1.reshape(-1, 64 * 40 * 40)
        x2 = x2.reshape(-1, 128 * 20 * 20)
        x3 = x3.reshape(-1, 256 * 10 * 10)

        return torch.cat((x1, x2, x3), dim=1)
