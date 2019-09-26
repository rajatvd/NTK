import torch
from torch import optim, nn
import copy


class ZeroOutput(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.init_model = [copy.deepcopy(model).eval()]
        self.model = model

    def forward(self, inp):
        return self.model(inp) - self.init_model[0](inp)


class Scale(nn.Module):
    def __init__(self, model, alpha):
        super().__init__()
        self.model = model

    def forward(self, inp):
        return alpha*self.model(inp)


def gd(model, xdata, ydata, iters=100, lr=1e-3, alpha=1):
    opt = optim.SGD(model.parameters(), lr=lr)
    for i in range(iters):
        out = model(xdata)
        loss = 1/(alpha**2) * nn.MSELoss()(out, ydata)
        opt.zero_grad()
        loss.backward()
        opt.step()
