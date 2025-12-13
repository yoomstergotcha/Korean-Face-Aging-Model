import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

# =========================
# Age Embedding
# =========================
class AgeEmbed(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, age_norm):  # (B,)
        return self.mlp(age_norm.view(-1, 1))


# =========================
# FiLM Residual Block
# =========================
class ResBlockFiLM(nn.Module):
    def __init__(self, channels, emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

        self.norm1 = nn.InstanceNorm2d(channels)
        self.norm2 = nn.InstanceNorm2d(channels)

        self.film = nn.Linear(emb_dim, channels * 2)

    def forward(self, x, emb):
        gamma, beta = self.film(emb).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        h = self.norm1(self.conv1(x))
        h = gamma * h + beta
        h = F.relu(h, inplace=True)

        h = self.norm2(self.conv2(h))
        return x + h


# =========================
# Upsample Block
# =========================
class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(x)


# =========================
# ResNet18 Encoder
# =========================
class ResNet18Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        net = resnet18(pretrained=pretrained)

        self.c1 = nn.Sequential(net.conv1, net.bn1, net.relu)
        self.c2 = nn.Sequential(net.maxpool, net.layer1)
        self.c3 = net.layer2
        self.c4 = net.layer3
        self.c5 = net.layer4

    def forward(self, x):
        f1 = self.c1(x)
        f2 = self.c2(f1)
        f3 = self.c3(f2)
        f4 = self.c4(f3)
        f5 = self.c5(f4)

        return {
            "c1": f1,
            "c2": f2,
            "c3": f3,
            "c4": f4,
            "c5": f5,
        }


# =========================
# SAM-Lite Generator
# =========================
class SAMResNetFiLMGenerator(nn.Module):
    def __init__(self, age_emb_dim=128):
        super().__init__()

        self.age_emb = AgeEmbed(age_emb_dim)
        self.enc = ResNet18Encoder(pretrained=True)

        self.b1 = ResBlockFiLM(512, age_emb_dim)
        self.b2 = ResBlockFiLM(512, age_emb_dim)

        self.up4 = UpBlock(512, 256)
        self.r8 = ResBlockFiLM(256 + 256, age_emb_dim)

        self.up3 = UpBlock(512, 128)
        self.r16 = ResBlockFiLM(128 + 128, age_emb_dim)

        self.up2 = UpBlock(256, 64)
        self.r32 = ResBlockFiLM(64 + 64, age_emb_dim)

        self.up1 = UpBlock(128, 64)
        self.r64 = ResBlockFiLM(64 + 64, age_emb_dim)

        self.up0 = UpBlock(128, 64)

        self.out_img = nn.Sequential(
            nn.Conv2d(64, 3, 1),
            nn.Tanh()
        )

        self.landmark_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 136)
        )

    def forward(self, x, age_src, age_tgt, age_min=0., age_max=80.):
        age_norm = (age_tgt - age_min) / (age_max - age_min + 1e-8)
        age_emb = self.age_emb(age_norm)

        feats = self.enc(x)
        h = feats["c5"]

        h = self.b1(h, age_emb)
        h = self.b2(h, age_emb)

        h = self.up4(h)
        h = self.r8(torch.cat([h, feats["c4"]], dim=1), age_emb)

        h = self.up3(h)
        h = self.r16(torch.cat([h, feats["c3"]], dim=1), age_emb)

        h = self.up2(h)
        h = self.r32(torch.cat([h, feats["c2"]], dim=1), age_emb)

        h = self.up1(h)
        h = self.r64(torch.cat([h, feats["c1"]], dim=1), age_emb)

        h = self.up0(h)

        img = self.out_img(h)
        lm = self.landmark_head(h).view(-1, 68, 2)

        return img, lm
