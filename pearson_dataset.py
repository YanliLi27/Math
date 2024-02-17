import numpy as np
from sklearn.metrics import r2_score
import torch


def train_data(length:int, seq:int, serity:float=0.5):
    x = torch.unsqueeze(torch.randn(length), dim=1)
    for i in range(seq):
        x1 = torch.unsqueeze(torch.randn(length), dim=1)
        x = torch.concat((x, x1), dim=-1)
    # y =  x * x2  # 
    y = torch.empty(x.size())


    bias = serity* torch.randn(x.size())
    y = x + bias

    z = []
    m = []
    n = []
    avg_pear = []
    for i in range(length):
        pear = r2_score(y_true=x[i], y_pred=y[i])
        if pear >= 0.1 and pear < 0.8:
            z.append(pear)
            m.append(np.asarray(x[i]))
            n.append(np.asarray(y[i]))
            avg_pear.append(pear)

        
    z = np.asarray(z, dtype=np.float32)
    m = np.asarray(m, dtype=np.float32)
    n = np.asarray(n, dtype=np.float32)
    z = torch.from_numpy(z)
    m = torch.from_numpy(m)
    n = torch.from_numpy(n)
    print(m.shape)
    print(n.shape)
    print(z.shape)
    avg_p, std_p = np.mean(np.asarray(avg_pear)), np.std(np.asarray(avg_pear))
    print('avg pearson', avg_p, ' std:', std_p)
    # y = ((x-torch.mean(x)) * (x2-torch.mean(x2)))
    return m, n, torch.unsqueeze(z, dim=1)