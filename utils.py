import torch
from torch import optim, nn
import copy
from tqdm import tqdm
from sacred.observers import FileStorageObserver


class ZeroOutput(nn.Module):
    """Zero the output of a model by subtracting out a copy of it."""

    def __init__(self, model):
        super().__init__()
        self.init_model = [copy.deepcopy(model).eval()]

        self.model = model

    def forward(self, inp):
        return self.model(inp) - self.init_model[0](inp)


class Scale(nn.Module):
    """Scale the output of the model by alpha."""

    def __init__(self, model, alpha):
        super().__init__()
        self.model = model
        self.alpha = alpha

    def forward(self, inp):
        return self.alpha*self.model(inp)


ACTS = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh
}


def simple_net(width,
               bias=True,
               zero_output=True,
               alpha=1,
               hidden_layers=1,
               act='relu',
               **kwargs):
    """A simple 1d input to 1d output deep ReLU network.

    Parameters
    ----------
    bias : bool
        Whether to include biases.
    zero_output : bool
        Whether to zero the output of the model.
    alpha : float
        Scale of output.
    """
    a = ACTS[act]
    model = nn.Sequential(nn.Linear(1, width, bias=bias),
                          a(),
                          *[layer
                            for _ in range(hidden_layers-1)
                            for layer in [nn.Linear(width, width, bias=bias), a()]
                            ],
                          nn.Linear(width, 1, bias=bias))

    if zero_output:
        model = ZeroOutput(model)
    model = Scale(model, alpha)
    return model


def ntk(model, inp):
    """Calculate the neural tangent kernel of the model on the inputs.

    Returns the gradient feature map along with the tangent kernel.
    """
    out = model(inp)
    p_vec = nn.utils.parameters_to_vector(model.parameters())
    p, = p_vec.shape
    n, outdim = out.shape
    assert outdim == 1, "cant handle output dim higher than 1 for now"

    # this is the transpose jacobian (grad y(w))^T)
    features = torch.zeros(n, p, requires_grad=False)

    for i in range(n):  # for loop over data points
        model.zero_grad()
        out[i].backward(retain_graph=True)
        p_grad = torch.tensor([], requires_grad=False)
        for p in model.parameters():
            p_grad = torch.cat((p_grad, p.grad.reshape(-1)))
        features[i, :] = p_grad

    tk = features @ features.t()  # compute the tangent kernel
    return features, tk


def get_run_dir(ex):
    """Helper for sacred experiment logging."""
    for obs in ex.observers:
        if type(obs) == FileStorageObserver:
            return obs.dir

    return '.'


def gd(model, xdata, ydata,
       iters=100,
       lr=1e-3,
       alpha=1,
       save_every=-1,
       ex=None,
       run_dir='.',
       progress_bar=True,
       eps=1e-10):
    """Run gradient descent using square loss on the model with the given data.

    Updates the given model instance.

    Parameters
    ----------
    alpha : float
        Scaling factor to normalize by. The loss is divided by alpha^2.
    save_every : int
        Interval with which to save model instances.
    ex : Sacred experiment
        Experiment to use for logging and saving.
    run_dir : str
        Path of directory to save models.
    eps : float
        Stop if the loss reduces below this value.

    Returns
    -------
    list of loss values

    """

    opt = optim.SGD(model.parameters(), lr=lr)
    losses = []

    if ex != None:
        run_dir = get_run_dir(ex)

    litem = -1
    t = range(iters)
    if progress_bar:
        t = tqdm(t)
    for i in t:
        if save_every != -1 and i % save_every == 0:
            # torch.save(model.state_dict(),
            #            f"{run_dir}/{i:06d}_model_{litem:.4f}.statedict")
            fname = f"{run_dir}/{i:06d}_model_{litem:.4f}.model"
            torch.save(model.eval(), fname)
            # ex.add_artifact(fname, f"{i:06d}_model.model")

        out = model(xdata)

        # normalizing the loss:
        loss = 1/(alpha**2) * nn.MSELoss()(out, ydata)

        # we store the unnormalized losses
        litem = loss.item()*(alpha**2)
        losses.append(litem)
        if progress_bar:
            t.set_postfix(loss=litem)
        if ex != None:
            ex.log_scalar(litem, 'loss')

        if litem < eps:
            return losses
        opt.zero_grad()
        loss.backward()
        opt.step()

    return losses


def linear_gd(A, b, x0,
              iters=100,
              lr=1e-3,
              alpha=1,
              save_every=-1,
              ex=None,
              run_dir='.'):

    eps = 1e-10
    m, p = A.shape
    x = nn.Parameter(x0.clone())
    opt = optim.SGD([x], lr=lr)
    losses = []

    if ex != None:
        run_dir = get_run_dir(ex)

    litem = -1
    t = tqdm(range(iters))
    for i in t:
        if save_every != -1 and i % save_every == 0:
            # torch.save(model.state_dict(),
            #            f"{run_dir}/{i:06d}_model_{litem:.4f}.statedict")
            fname = f"{run_dir}/{i:06d}_weight_{litem:.4f}.parameter"
            torch.save(x, fname)
            # ex.add_artifact(fname, f"{i:06d}_model.model")

        out = A @ (x - x0)
        # print(out.shape, b.shape)
        loss = 1/(alpha**2) * nn.MSELoss()(out.squeeze(), b)
        litem = loss.item()*(alpha**2)
        losses.append(litem)
        t.set_postfix(loss=litem)
        if ex != None:
            ex.log_scalar(litem, 'loss')

        if litem < eps:
            return losses
        opt.zero_grad()
        loss.backward()
        opt.step()

    return losses
