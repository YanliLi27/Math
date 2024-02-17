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
        # x [batch, 3, (1)]  # [batch, 3, (1)]
        x = torch.concat((x, y), dim=-1)
        x = self.features(x)
        x = self.attend(x)
        return x