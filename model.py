import torch
import torch.nn as nn
import torch.nn.functional as F

class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        self.cv1 = ConvBnAct(c1, c1 // 2, k=1)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv2 = ConvBnAct((c1 // 2) * 4, c2, k=1)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))

class ConvBnAct(nn.Sequential):
    def __init__(self, c1, c2, k=1, s=1, p=0, act=True):
        layers = [nn.Conv2d(c1, c2, k, s, p, bias=False),
                  nn.BatchNorm2d(c2)]
        if act:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)

class C2f(nn.Module):
    def __init__(self, c1, c2, n=1):
        super().__init__()
        self.cv1 = ConvBnAct(c1, c2 // 2, k=1)
        self.cv2 = ConvBnAct(c1, c2 // 2, k=1)
        self.m = nn.Sequential(*[
            ConvBnAct(c2 // 2, c2 // 2, k=3) for _ in range(n)
        ])
        self.cv3 = ConvBnAct(c2, c2, k=1)

    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        if y1.shape[2:] != y2.shape[2:]:
            y2 = F.interpolate(y2, size=y1.shape[2:], mode='nearest')
        return self.cv3(torch.cat([y1, y2], dim=1))


class PANNeck(nn.Module):
    def __init__(self, channels_list):
        super().__init__()
        # Top-down
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, c // 2, 1) for c in channels_list
        ])
        self.fpn_convs = nn.ModuleList([
            C2f(c // 2, c // 2, n=1) for c in channels_list
        ])
        # Bottom-up
        self.down_convs = nn.ModuleList([
            nn.Conv2d(c // 2, c // 2, 3, 2, 1) for c in channels_list[:-1]
        ])
        self.pan_convs = nn.ModuleList([
            C2f(c // 2, c // 2, n=1) for c in channels_list
        ])

        # Align channels on upsampling (top-down)
        self.top_align = nn.ModuleList([
            ConvBnAct(channels_list[i+1] // 2, channels_list[i] // 2, k=1)
            for i in range(len(channels_list)-1)
        ])
        # Align channels on downsampling (bottom-up)
        self.bottom_align = nn.ModuleList([
            ConvBnAct(channels_list[i] // 2, channels_list[i+1] // 2, k=1)
            for i in range(len(channels_list)-1)
        ])

    def forward(self, features):
        # Top-down FPN
        fpn_outs, last = [], None
        for i, feat in enumerate(reversed(features)):
            idx = len(features)-1 - i
            lat = self.lateral_convs[idx](feat)
            if last is not None:
                up = F.interpolate(last, scale_factor=2, mode='nearest')
                if idx < len(self.top_align):
                    up = self.top_align[idx](up)
                if up.shape[2:] != lat.shape[2:]:
                    up = F.interpolate(up, size=lat.shape[2:], mode='nearest')
                lat = lat + up
            last = self.fpn_convs[idx](lat)
            fpn_outs.insert(0, last)

        # Bottom-up PAN
        pan_outs = [fpn_outs[0]]
        for i in range(len(fpn_outs)-1):
            down = self.down_convs[i](pan_outs[i])
            # align channels to match fpn_outs[i+1]
            down = self.bottom_align[i](down)
            # align spatial to match fpn_outs[i+1]
            if down.shape[2:] != fpn_outs[i+1].shape[2:]:
                down = F.interpolate(down, size=fpn_outs[i+1].shape[2:], mode='nearest')
            fusion = down + fpn_outs[i+1]
            pan_outs.append(self.pan_convs[i+1](fusion))

        return pan_outs

# Detect Head (single conv per scale) with DFL
class DetectHead(nn.Module):
    def __init__(self, channels_list, nc=1, reg_max=16):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.no = nc + reg_max * 4
        self.detect_convs = nn.ModuleList([
            nn.Conv2d(c, self.no, kernel_size=1) for c in channels_list
        ])

    def forward(self, features):
        return [conv(feat) for conv, feat in zip(self.detect_convs, features)]


# Full official YOLOv8n for single class
class YOLOv8n(nn.Module):
    def __init__(self, nc=1, reg_max=16):
        super().__init__()

        # Backbone (depths n=1,2,3,1 with channel widths)
        self.stem = nn.Sequential(
            ConvBnAct(3, 64, k=3, s=2)
        )
        self.c2f1 = C2f(64, 128, n=1)
        self.down1 = ConvBnAct(128, 128, k=3, s=2)
        self.c2f2 = C2f(128, 256, n=2)
        self.down2 = ConvBnAct(256, 256, k=3, s=2)
        self.c2f3 = C2f(256, 512, n=3)
        self.down3 = ConvBnAct(512, 512, k=3, s=2)
        self.c2f4 = C2f(512, 1024, n=1)
        self.sppf = SPPF(1024, 1024)

        # Neck & Head channels (official)
        self.pan = PANNeck([256, 512, 1024])
        self.detect = DetectHead([128, 256, 512], nc=nc, reg_max=reg_max)

        # Model list for compatibility
        self.model = nn.ModuleList([
            self.stem, self.c2f1, self.down1,
            self.c2f2, self.down2, self.c2f3,
            self.down3, self.c2f4, self.sppf,
            self.pan, self.detect
        ])

        self.model[-1].stride = torch.tensor([8, 16, 32], dtype=torch.float32)
        self.model[-1].nc = nc
        self.model[-1].reg_max = reg_max
        self.args = {"box": 7.5, "cls": 0.5, "dfl": 1.5}

    def forward(self, x):
        x1 = self.c2f1(self.stem(x))
        x2 = self.c2f2(self.down1(x1))
        x3 = self.c2f3(self.down2(x2))
        x4 = self.c2f4(self.down3(x3))
        x4 = self.sppf(x4)
        features = self.pan([x2, x3, x4])
        return self.detect(features)