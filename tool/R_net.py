# !/usr/bin/python
# -*- coding:utf-8 -*-
import torch.nn as nn


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class R_net(nn.Module):
    def __init__(self, ):
        super(R_net, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(3, 28, kernel_size=3, stride=1),
            nn.PReLU(28),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.Conv2d(28, 48, kernel_size=3, stride=1),
            nn.PReLU(48),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.Conv2d(48, 64, kernel_size=2, stride=1),
            nn.PReLU(64),
            Flatten(),
            nn.Linear(576, 128),
            nn.PReLU(128),

        )

        self.logits = nn.Linear(128, 2)
        self.loc = nn.Linear(128, 4)
        self.landmarks = nn.Linear(128, 10)

    def forward(self, x):
        x = self.head(x)
        logits = self.logits(x)
        loc = self.loc(x)
        landmarks = self.landmarks(x)
        return logits, loc, landmarks
