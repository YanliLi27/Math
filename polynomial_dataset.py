import numpy as np
from sklearn.metrics import r2_score
import torch


def train_data(length:int, seq:int):
    x = torch.unsqueeze(torch.randn(length), dim=1)
    y = torch.unsqueeze(torch.randn(length), dim=1)
    for i in range(seq):
        x1 = torch.unsqueeze(torch.randn(length), dim=1)
        x = torch.concat((x, x1), dim=-1)
        y1 = torch.unsqueeze(torch.randn(length), dim=1)
        y = torch.concat((y, y1), dim=-1)

    z = []
    m = []
    n = []

    for i in range(length):
        if y[i].any():
            pear = x[i] / y[i]
            z.append(pear)
            m.append(np.asarray(x[i]))
            n.append(np.asarray(y[i]))
        
    z = np.asarray(z, dtype=np.float32)
    m = np.asarray(m, dtype=np.float32)
    n = np.asarray(n, dtype=np.float32)
    z = torch.from_numpy(z)
    m = torch.from_numpy(m)
    n = torch.from_numpy(n)
    print(m.shape)
    print(n.shape)
    print(z.shape)
    # y = ((x-torch.mean(x)) * (x2-torch.mean(x2)))
    return m, n, torch.unsqueeze(z, dim=1)