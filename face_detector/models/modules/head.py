import torch
import torch.nn as nn


class HeadYoloV5N(nn.Module):
    def __init__(self, n_classes, anchors, width_multiple):
        super(HeadYoloV5N, self).__init__()
        self.n_anchor = len(anchors)
        self.n_output = n_classes + 4 + 1  # 4: x, y, w, h; 1: object ness score
        self.conv1 = nn.Conv2d(int(128 * width_multiple), self.n_output * self.n_anchor, 1)
        self.conv2 = nn.Conv2d(int(128 * width_multiple), self.n_output * self.n_anchor, 1)
        self.conv3 = nn.Conv2d(int(128 * width_multiple), self.n_output * self.n_anchor, 1)

    def forward(self, features):
        x = features[:, :64 * 40 * 40].reshape(-1, 64, 40, 40).contiguous()
        y = features[:, 64 * 40 * 40:64 * 40 * 40 + 64 * 20 * 20].reshape(-1, 64, 20, 20).contiguous()
        z = features[:, 64 * 40 * 40 + 64 * 20 * 20:].reshape(-1, 64, 10, 10).contiguous()
        x = self.conv1(x).reshape(-1, 18 * 40 * 40)
        y = self.conv2(y).reshape(-1, 18 * 20 * 20)
        z = self.conv3(z).reshape(-1, 18 * 10 * 10)
        outputs = torch.cat((x, y, z), dim=1)
        return outputs


class DecoderYoloV5(nn.Module):
    def __init__(self, n_classes, anchors, stride):
        super(DecoderYoloV5, self).__init__()
        self.nc = n_classes
        self.no = self.nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2

        self.anchors_grid = anchors.float().view(3, -1, 2)
        self.anchors = self.anchors_grid / stride.view(-1, 1, 1)
        self.stride = stride

    def _decode(self, output, idx, anchor_grid):
        batch_size, n_anchor, h, w, n_output = output.shape
        grid = self._make_grid(w, h)
        y = torch.full_like(output, 0)
        # class_range = torch.tensor(list(range(0, 5 + self.nc)))
        class_range = list(range(0, 5 + self.nc))
        y[..., class_range] = output[..., class_range].sigmoid()
        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid.to(output.device)) * self.stride[idx]  # xy
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[idx].to(output.device)  # wh

        return y.view(batch_size, -1, n_output)

    def forward(self, outputs):
        a = outputs[:, 0:18 * 40 * 40].reshape(1, 18, 40, 40)
        a = a.view(1, 3, 6, 40, 40).permute(0, 1, 3, 4, 2).contiguous()
        b = outputs[:, 18 * 40 * 40:18 * 40 * 40 + 18 * 20 * 20].reshape(1, 18, 20, 20)
        b = b.view(1, 3, 6, 20, 20).permute(0,1,3,4,2).contiguous()
        c = outputs[:, 18 * 40 * 40 + 18 * 20 * 20:].reshape(1, 18, 10, 10)
        c = c.view(1, 3, 6, 10, 10).permute(0, 1, 3, 4, 2).contiguous()

        anchor_grid = self.anchors_grid.view(3, 1, -1, 1, 1, 2)
        feature_map_a = self._decode(a, 0, anchor_grid)
        feature_map_b = self._decode(b, 1, anchor_grid)
        feature_map_c = self._decode(c, 2, anchor_grid)

        return torch.cat((feature_map_a, feature_map_b, feature_map_c), dim=1)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
