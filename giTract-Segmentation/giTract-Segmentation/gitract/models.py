"""AGTSNet: residual encoder with squeeze-and-excitation, ASPP bottleneck, attention-gated skip connections.

This file follows the architecture described in the paper (Fig. 2). If you hold the original model definition,
replace this module with it; the rest of the pipeline only requires `AGTSNet(in_ch, n_classes)` returning logits.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_relu(i, o, k=3, d=1):
    return nn.Sequential(nn.Conv2d(i, o, k, padding=d * (k // 2), dilation=d, bias=False),
                         nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class SE(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(c, max(c // r, 4)), nn.ReLU(inplace=True),
                                nn.Linear(max(c // r, 4), c), nn.Sigmoid())

    def forward(self, x):
        w = self.fc(x.mean(dim=(2, 3)))
        return x * w[:, :, None, None]


class ResSE(nn.Module):
    def __init__(self, i, o, stride=1):
        super().__init__()
        self.c1 = conv_bn_relu(i, o)
        self.c2 = nn.Sequential(nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o))
        self.se = SE(o)
        self.skip = nn.Sequential(nn.Conv2d(i, o, 1, bias=False), nn.BatchNorm2d(o))
        self.stride = stride

    def forward(self, x):
        if self.stride > 1:
            x = F.max_pool2d(x, self.stride)
        y = self.se(self.c2(self.c1(x)))
        return F.relu(y + self.skip(x))


class ASPP(nn.Module):
    def __init__(self, i, o, rates=(6, 12, 18)):
        super().__init__()
        self.b0 = conv_bn_relu(i, o, 1)
        self.bs = nn.ModuleList([conv_bn_relu(i, o, 3, d=r) for r in rates])
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(i, o, 1, bias=False), nn.ReLU(inplace=True))
        self.proj = conv_bn_relu(o * (len(rates) + 2), o, 1)

    def forward(self, x):
        outs = [self.b0(x)] + [b(x) for b in self.bs]
        outs.append(F.interpolate(self.pool(x), size=x.shape[2:], mode="bilinear", align_corners=False))
        return self.proj(torch.cat(outs, dim=1))


class AttentionGate(nn.Module):
    def __init__(self, c_skip, c_gate, c_int):
        super().__init__()
        self.wx = nn.Conv2d(c_skip, c_int, 1)
        self.wg = nn.Conv2d(c_gate, c_int, 1)
        self.psi = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(c_int, 1, 1), nn.Sigmoid())

    def forward(self, skip, gate):
        g = F.interpolate(self.wg(gate), size=skip.shape[2:], mode="bilinear", align_corners=False)
        return skip * self.psi(self.wx(skip) + g)


class Up(nn.Module):
    def __init__(self, c_in, c_skip, c_out):
        super().__init__()
        self.ag = AttentionGate(c_skip, c_in, c_skip // 2)
        self.conv = nn.Sequential(conv_bn_relu(c_in + c_skip, c_out), conv_bn_relu(c_out, c_out))

    def forward(self, x, skip):
        up = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([up, self.ag(skip, x)], dim=1))


class AGTSNet(nn.Module):
    def __init__(self, in_ch=1, n_classes=3, widths=(64, 128, 256, 512)):
        super().__init__()
        w = widths
        self.stem = conv_bn_relu(in_ch, w[0])
        self.e1 = ResSE(w[0], w[0])
        self.e2 = ResSE(w[0], w[1], stride=2)
        self.e3 = ResSE(w[1], w[2], stride=2)
        self.e4 = ResSE(w[2], w[3], stride=2)
        self.aspp = ASPP(w[3], w[3])
        self.u3 = Up(w[3], w[2], w[2])
        self.u2 = Up(w[2], w[1], w[1])
        self.u1 = Up(w[1], w[0], w[0])
        self.head = nn.Conv2d(w[0], n_classes, 1)   # logits; sigmoid is applied in the loss / at inference

    def forward(self, x):
        s1 = self.e1(self.stem(x))
        s2 = self.e2(s1)
        s3 = self.e3(s2)
        b = self.aspp(self.e4(s3))
        d = self.u1(self.u2(self.u3(b, s3), s2), s1)
        return self.head(d)
