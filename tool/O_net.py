# !/usr/bin/python
# -*- coding:utf-8 -*-
import torch.nn as nn


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        # x = x.transpose(3, 2).contiguous()
        return x.view(x.size(0), -1)


class O_net(nn.Module):
    def __init__(self, ):
        super(O_net, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),
            nn.PReLU(32),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.PReLU(64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.PReLU(64),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),

            nn.Conv2d(64, 128, kernel_size=2, stride=1),
            nn.PReLU(128),

            Flatten(),
            nn.Linear(1152, 256),
            nn.Dropout(0.25),
            nn.PReLU(256)

        )

        self.logits = nn.Linear(256, 2)
        self.loc = nn.Linear(256, 4)
        self.landmarks = nn.Linear(256, 10)

    def forward(self, x):
        x = self.head(x)
        logits = self.logits(x)
        loc = self.loc(x)
        landmarks = self.landmarks(x)
        return logits, loc, landmarks
