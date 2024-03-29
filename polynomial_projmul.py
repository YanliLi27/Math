import torch
import matplotlib.pyplot as plt
import torch.nn as nn


class NN(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output, depth=1
                 ):
        super().__init__()
        self.hidden = [nn.Linear(n_feature, n_hidden), nn.ReLU()]
        if depth > 1:
            in_ch = n_hidden
            out_ch = 2*n_hidden
            for _ in range(depth-1):
                self.hidden.append(nn.Linear(in_ch, out_ch)) # [batch, 1, (1)])
                self.hidden.append(nn.ReLU())
                in_ch = out_ch
                out_ch = 2* out_ch
        self.features = nn.Sequential(*self.hidden)
        self.attend = nn.Linear(out_ch//2, n_output)


    def forward(self, x, y):
        # x [batch, 1, (1)]  # [batch, 1, (1)]
        x = torch.concat((x, y), dim=-1)
        x = self.features(x)
        x = self.attend(x)
        return x

    
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.unsqueeze(torch.linspace(1, 100, 10000), dim=1)
x2 = torch.unsqueeze(torch.linspace(1, 100, 10000), dim=1)
y =  x / x2  # 
# y = (x * x2)/ (torch.std(x)*torch.std(x2))
# y = ((x-torch.mean(x)) * (x2-torch.mean(x2)))
print(x.shape)
print(y.shape)
print(torch.concat((x, x2), dim=-1).shape)

x, x2, y = x.to(device), x2.to(device), y.to(device)
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

net = NN(2, 10, 1, 4)
net = net.to(device)
# optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
optimizer = torch.optim.AdamW(net.parameters(), lr=0.01)

lossfunc = torch.nn.MSELoss()


tx = torch.unsqueeze(torch.linspace(1, 120, 1000), dim=1)
tx2 = torch.unsqueeze(torch.linspace(1, 120, 1000), dim=1)
yt = tx * tx2
tx, tx2 = tx.to(device), tx2.to(device)


for i in range(1000):
    pred = net(x, x2)
    loss = lossfunc(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    if i%5==0:
        plt.cla()
        plt.scatter(x.cpu().data.numpy(), y.cpu().data.numpy())
        plt.plot(x.cpu().data.numpy(), pred.cpu().data.numpy(), 'r-', lw=5)
        plt.pause(0.1)

print('end')
t_pred = net(tx, tx2).cpu().data.numpy()
plt.cla()
plt.scatter(x.cpu().data.numpy(), y.cpu().data.numpy())
plt.plot(x.cpu().data.numpy(), pred.cpu().data.numpy(), 'r-', lw=5)
plt.scatter(tx.cpu().data.numpy(), yt)
plt.plot(tx.cpu().data.numpy(), t_pred, 'b-', lw=5)
plt.show()








