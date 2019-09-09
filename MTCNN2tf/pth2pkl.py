# !/usr/bin/python
# -*- coding:utf-8 -*-
import torch
from sklearn.externals import joblib
if __name__ == "__main__":
    P = []
    x = torch.load('../train/models/P_net_45000_1.pth')
    for k in x.keys():
        v = x[k]
        print(k, v.shape)
        if len(v.shape) == 4:
            v = v.permute(2, 3, 1, 0)

        v = v.cpu().numpy()
        P.append(v)
    joblib.dump(P, './models/P.pkl')

    print('===============================')

    P = []
    x = torch.load('../train/models/R_net_45000_1.pth')
    for k in x.keys():
        v = x[k]
        print(k, v.shape)
        if len(v.shape) == 4:
            v = v.permute(2, 3, 1, 0)
        if len(v.shape) == 2:
            v = v.permute(1, 0)

        v = v.cpu().numpy()
        P.append(v)
    joblib.dump(P, './models/R.pkl')
    print('===============================')
    P = []
    x = torch.load('../train/models/O_net_45000_1.pth')
    for k in x.keys():
        v = x[k]
        print(k, v.shape)
        if len(v.shape) == 4:
            v = v.permute(2, 3, 1, 0)
        if len(v.shape) == 2:
            v = v.permute(1, 0)

        v = v.cpu().numpy()
        P.append(v)
    joblib.dump(P, './models/O.pkl')
