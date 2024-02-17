import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
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


    def forward(self, x):
        # x [batch, 1, (1)]
        x = self.features(x)
        x = self.attend(x)
        return x

    

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
x2 = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y =  x.pow(-2)  # x * x2  # 
print(x.shape)
print(y.shape)

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

net = NN(1, 10, 1, 4)
print(net)
# optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
optimizer = torch.optim.AdamW(net.parameters(), lr=0.01)

lossfunc = torch.nn.MSELoss()

for i in range(1000):
    pred = net(x)
    loss = lossfunc(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    if i%5==0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), pred.data.numpy(), 'r-', lw=5)
        plt.pause(0.1)







