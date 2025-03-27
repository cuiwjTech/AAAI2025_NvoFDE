import torch
import torch.nn.functional as F
import warnings
from torch import nn

def _check_inputs(a1, b1, func, y0, t, step_size,method,beta, SOLVERS):








    if method not in SOLVERS:
        raise ValueError('Invalid method "{}". Must be one of {}'.format(method,
                                                                         '{"' + '", "'.join(SOLVERS.keys()) + '"}.'))




    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float32, device=y0.device)

    else:
        t = t.to(y0.device)

    if not (t > 0).all():
        raise ValueError("t must be > 0")

    if not isinstance(beta, torch.Tensor):
        beta = torch.tensor(beta, dtype=torch.float32, device=y0.device)

    else:
        beta = beta.to(y0.device)

    positive_beta = F.relu(beta)
    negative_beta = torch.sigmoid(beta)
    beta = torch.where(beta > 0, positive_beta, negative_beta).to(dtype=torch.float32, device=y0.device)

    if not (beta <= 1).all():
        warnings.warn("beta should be <= 1 for the initial value problem")


    if not isinstance(step_size, torch.Tensor):
        step_size = torch.tensor(step_size, dtype=torch.float32, device=y0.device)

    else:
        step_size = step_size.to(y0.device)

    if not (step_size > 0).all():
        raise ValueError("step_size must be > 0")


    if not (step_size < t).all():
        raise ValueError("step_size must be < t")
    tspan = torch.arange(0,t,step_size)




    return a1, b1, func, y0, tspan, method, beta
