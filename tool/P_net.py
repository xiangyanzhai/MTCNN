# !/usr/bin/python
# -*- coding:utf-8 -*-
import torch.nn as nn


class P_net(nn.Module):
    def __init__(self, ):
        super(P_net, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3, stride=1),
            nn.PReLU(10),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Conv2d(10, 16, 3, 1),
            nn.PReLU(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.PReLU(32),

        )

        self.logits = nn.Conv2d(32, 2, kernel_size=1, stride=1)
        self.loc = nn.Conv2d(32, 4, kernel_size=1, stride=1)
        self.landmarks = nn.Conv2d(32, 10, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.head(x)
        logits = self.logits(x)
        loc = self.loc(x)
        landmarks = self.landmarks(x)
        return logits, loc, landmarks
