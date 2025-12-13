import torch
import torch.nn as nn

class SSRNet(nn.Module):
    def __init__(self, stage_num=[3,3,3], lambda_local=0.25, lambda_d=0.5, age=True):
        super().__init__()
        self.stage_num = stage_num
        self.lambda_local = lambda_local
        self.lambda_d = lambda_d
        self.age = age

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        feat = self.backbone(x).view(x.size(0), -1)
        age = self.fc(feat).squeeze(1)
        return age
