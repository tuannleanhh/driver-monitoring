import torch
import torch.nn as nn

from .modules import BackboneYoloV5N, NeckYoloV5N, HeadYoloV5N, DecoderYoloV5


class YoloV5(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, anchors=None, stride=(8, 16, 32), width_multiple=0.5):
        super(YoloV5, self).__init__()
        self.stride = torch.tensor(stride)
        self.names = [str(i) for i in range(num_classes)]
        self.nc = num_classes
        self.anchors = anchors if anchors is not None else torch.tensor([[4, 5, 8, 10, 13, 16],
                                                                        [23, 29, 43, 55, 73, 105],
                                                                        [146, 217, 231, 300, 335, 433]])
        self.backbone = BackboneYoloV5N(in_channels, width_multiple)
        self.neck = NeckYoloV5N(width_multiple)
        self.head = HeadYoloV5N(num_classes, self.anchors, width_multiple)
        self.decoder = DecoderYoloV5(num_classes, self.anchors, self.stride)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return self.decoder(x)
