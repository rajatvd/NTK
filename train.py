from sacred import Experiment
from utils import Scale, ZeroOutput, gd, simple_net, ntk, linear_gd
from torch import nn
import torch

ex = Experiment("toy-example")


@ex.config
def config():
    width = 100
    hidden_layers = 1
    act = 'relu'
    bias = True
    alpha = 1
    zero_output = True
    lr = 1e-3
    iters = 10000
    save_every = 100
    xdata = [-0.1, 0.5]
    ydata = [0.4, 0.1]
    linearize = False  # if true, train the linearized model around initialization


@ex.automain
def main(width,
         hidden_layers,
         act,
         bias,
         alpha,
         zero_output,
         xdata,
         ydata,
         lr,
         iters,
         save_every,
         linearize,
         _run):

    model = simple_net(width=width,
                       bias=bias,
                       zero_output=zero_output,
                       alpha=alpha,
                       act=act,
                       hidden_layers=hidden_layers,)

    xdata = torch.tensor(xdata).float().unsqueeze(1)
    ydata = torch.tensor(ydata).float().unsqueeze(1)

    if linearize:
        A, tk = ntk(model, xdata)
        w0 = nn.utils.parameters_to_vector(model.parameters()).detach().numpy()
        w0 = torch.tensor(w0.copy(), requires_grad=False)
        losses = linear_gd(A, ydata.squeeze(), w0,
                           iters=iters,
                           lr=lr,
                           alpha=alpha,
                           ex=_run,
                           save_every=save_every)
    else:
        losses = gd(model, xdata, ydata,
                    iters=iters,
                    lr=lr,
                    alpha=alpha,
                    ex=_run,
                    save_every=save_every)
