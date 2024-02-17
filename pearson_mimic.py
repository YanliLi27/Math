import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from pearson_dataset import train_data
from pearson_model import NN


seq = 29
serity = 0.4
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
m, n, z = train_data(10000, seq, serity)
m, n, z = m.to(device), n.to(device), z.to(device)

mt, nt, zt = train_data(1000, seq, serity)
mt, nt, zt = mt.to(device), nt.to(device), zt.to(device)

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

net = NN(2*seq+2, 2*seq+2, 1, 8)

net = net.to(device)

# print(net)
# optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)

lossfunc = torch.nn.MSELoss()

for i in range(1000):
    for b in range(n.shape[0]//2000):
        pred = net(n[b*2000:b*2000+2000], m[b*2000:b*2000+2000])
        loss = lossfunc(pred, z[b*2000:b*2000+2000])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    t_pred = net(nt, mt)
    loss_t = lossfunc(t_pred, zt)

    if i%10==0:
        print(f'train loss: {str(loss.item())[:10]}, test: {str(loss_t.item())[:10]}')
