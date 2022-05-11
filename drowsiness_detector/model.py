import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        super(ConvBNReLU, self).__init__()
        padding = (kernel_size - 1) // 2
        self.Conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False)
        self.BN = nn.BatchNorm2d(out_planes)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Conv(x)
        x = self.BN(x)
        x = self.ReLU(x)

        return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, outp, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(expand_ratio * inp))
        self.use_res_connect = stride == 1 and inp == outp
        self.expand_ratio = expand_ratio

        self.conv_bn_relu = ConvBNReLU(inp, hidden_dim, kernel_size=1)
        self._dw_conv_bn_relu = ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim)
        self._pw_conv = nn.Conv2d(hidden_dim, outp, 1, 1, 0, bias=False)
        self._pw_BN = nn.BatchNorm2d(outp)

    def forward(self, _input):
        if self.expand_ratio != 1:
            x = self.conv_bn_relu(_input)
        else:
            x = _input
        x = self._dw_conv_bn_relu(x)
        x = self._pw_conv(x)
        x = self._pw_BN(x)

        if (self.use_res_connect):
            return x + _input
        else:
            return x


class InvertedResidual1(nn.Module):

    def __init__(self, inp, outp, stride, expand_ratio):
        super(InvertedResidual1, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(expand_ratio * inp))
        self._dw_conv_bn_relu = ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim)
        self._pw_conv = nn.Conv2d(hidden_dim, outp, 1, 1, 0, bias=False)
        self._pw_BN = nn.BatchNorm2d(outp)

    def forward(self, x):
        x = self._dw_conv_bn_relu(x)
        x = self._pw_conv(x)
        x = self._pw_BN(x)

        return x


class MobileNetV2(nn.Module):

    def __init__(self, num_classes=2):
        super(MobileNetV2, self).__init__()
        # First conv bn relu block
        self._conv_bn_relu_1 = ConvBNReLU(in_planes=3, out_planes=32, kernel_size=3, stride=2, groups=1)

        # Inverted residual blocks
        self._inverted_residual_1 = InvertedResidual1(inp=32, outp=16, stride=1, expand_ratio=1)
        self._inverted_residual_2 = InvertedResidual(inp=16, outp=24, stride=2, expand_ratio=6)
        self._inverted_residual_3 = InvertedResidual(inp=24, outp=24, stride=1, expand_ratio=6)
        self._inverted_residual_4 = InvertedResidual(inp=24, outp=32, stride=2, expand_ratio=6)
        self._inverted_residual_5 = InvertedResidual(inp=32, outp=32, stride=1, expand_ratio=6)
        self._inverted_residual_6 = InvertedResidual(inp=32, outp=32, stride=1, expand_ratio=6)
        self._inverted_residual_7 = InvertedResidual(inp=32, outp=64, stride=2, expand_ratio=6)
        self._inverted_residual_8 = InvertedResidual(inp=64, outp=64, stride=1, expand_ratio=6)
        self._inverted_residual_9 = InvertedResidual(inp=64, outp=64, stride=1, expand_ratio=6)
        self._inverted_residual_10 = InvertedResidual(inp=64, outp=64, stride=1, expand_ratio=6)
        self._inverted_residual_conv_bn_relu = ConvBNReLU(in_planes=64, out_planes=1280, kernel_size=1, stride=1,
                                                          groups=1)
        # Classifier block
        self.drop_out = nn.Dropout(p=0.2, inplace=False)
        self.linear = nn.Linear(1280, num_classes)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, _input):
        x = self._conv_bn_relu_1(_input)
        x = self._inverted_residual_1(x)
        x = self._inverted_residual_2(x)
        x = self._inverted_residual_3(x)
        x = self._inverted_residual_4(x)
        x = self._inverted_residual_5(x)
        x = self._inverted_residual_6(x)
        x = self._inverted_residual_7(x)
        x = self._inverted_residual_8(x)
        x = self._inverted_residual_9(x)
        x = self._inverted_residual_10(x)
        x = self._inverted_residual_conv_bn_relu(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.drop_out(x)
        x = self.linear(x)

        return x
