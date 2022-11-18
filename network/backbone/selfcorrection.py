import torch
import torch.nn as nn


class SelfCorrectionModule(nn.Module):
    def __init__(self, in_classes=21, num_features=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_classes*2, num_features, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_features, in_classes, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_classes)
        self.relu2 = nn.ReLU(inplace=True)

        self._init_weight()

    def forward(self, primary_logits, ancillary_logits):
        x = torch.cat([primary_logits, ancillary_logits], dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        out = self.relu2(x)

        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
