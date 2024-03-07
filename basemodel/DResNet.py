import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, kernel_size=3) -> None:
        super(ResBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding='same' if stride ==1 else 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if inchannel != outchannel or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, kernel_size=3):
        super(ResNet18, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(5, 64, kernel_size=3, stride=1, padding="same", bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResBlock, 64, 2, stride=1, kernel_size=kernel_size)
        self.layer2 = self.make_layer(ResBlock, 128, 2, stride=1, kernel_size=kernel_size)
        self.layer3 = self.make_layer(ResBlock, 256, 2, stride=1, kernel_size=kernel_size)
        self.layer4 = self.make_layer(ResBlock, 512, 2, stride=1, kernel_size=kernel_size)   
        # self.fc = nn.Linear(512, 3)

    def make_layer(self, block, channels, num_blocks, stride, kernel_size):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride, kernel_size))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.to(torch.float32)
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out
    
    def output_dim(self):
        return 2048
