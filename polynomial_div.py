import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from polynomial_dataset import div_data
from pearson_model import NN

    
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x, y, z = div_data(length=10000, seq=0)
x, y, z = x.to(device), y.to(device), z.to(device)
# [batch=length, channel=seq]

net = NN(2, 10, 1, 8)
net = net.to(device)
# optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)

lossfunc = torch.nn.MSELoss()

tx, ty, tz = div_data(length=1000, seq=0)
tx, ty, tz = tx.to(device), ty.to(device), tz.to(device)

for i in range(1000):
    pred = net(x, y)
    loss = lossfunc(pred, z)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    t_pred = net(tx, ty)
    loss_t = lossfunc(t_pred, tz)

    if i%10==0:
        print(f'train loss: {str(loss.item())[:10]}, test: {str(loss_t.item())[:10]}')








