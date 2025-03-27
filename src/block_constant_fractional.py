from base_classes import ODEblock
import torch
from torch import nn
from utils import get_rw_adj, gcn_norm_fill_val
from torchfde import fdeint, fdeint1, fdeint2



  




class ConstantODEblock_FRAC2(ODEblock):
  def __init__(self, odefunc,  opt, data,  device, t=torch.tensor([0, 1])):
    super(ConstantODEblock_FRAC2, self).__init__(odefunc,  opt, data,   device, t)

    self.odefunc = odefunc(opt['hidden_dim'], opt['hidden_dim'], opt, data, device)
    if opt['data_norm'] == 'rw':
      edge_index, edge_weight = get_rw_adj(data.edge_index, edge_weight=data.edge_attr, norm_dim=1,
                                                                   fill_value=opt['self_loop_weight'],
                                                                   num_nodes=data.num_nodes,
                                                                   dtype=data.x.dtype)
    else:
      edge_index, edge_weight = gcn_norm_fill_val(data.edge_index, edge_weight=data.edge_attr,
                                           fill_value=opt['self_loop_weight'],
                                           num_nodes=data.num_nodes,
                                           dtype=data.x.dtype)
    self.odefunc.edge_index = edge_index.to(device)
    self.odefunc.edge_weight = edge_weight.to(device)

    if opt['adjoint']:
      from torchdiffeq import odeint_adjoint as odeint
    else:
      from torchdiffeq import odeint

    self.train_integrator = odeint
    self.test_integrator = odeint
    self.device = device
    self.opt = opt

    self.fc0 = nn.Sequential(nn.Linear(opt['hidden_dim'], 2 * opt['hidden_dim']),
                             nn.ReLU(),
                             nn.Linear(2 * opt['hidden_dim'], 1)).to(device)
    self.fc1 = nn.Linear(opt['hidden_dim'], opt['hidden_dim']).to(device)

    if isinstance(self.opt['alpha_ode'], float):
      alpha_ode = [self.opt['alpha_ode']]
    else:
      alpha_ode = self.opt['alpha_ode']
    alpha_ode = torch.tensor(alpha_ode, device=device)
    tspan = torch.arange(0, self.opt['time'], self.opt['step_size'])
    N1 = len(tspan)
    self.alpha = nn.Parameter(alpha_ode.repeat(N1).clone().detach().to(device), requires_grad=True)
    self.a1 = nn.Parameter(torch.ones(1))
    self.b1 = nn.Parameter(torch.ones(1))
    self.c1 = nn.Parameter(torch.ones(1))
    self.d1 = nn.Parameter(torch.ones(1))
    
  def forward(self, x):
    t = self.t.type_as(x)

    integrator = self.train_integrator if self.training else self.test_integrator


    func = self.odefunc
    state = x


    mask_out_of_bounds = (self.alpha.data <= 0) | (self.alpha.data > 1)
    # if mask_out_of_bounds.any():
    #     self.alpha.data[mask_out_of_bounds] = torch.abs(torch.sin(1 / 2 * math.pi * self.alpha.data[mask_out_of_bounds]))
    if mask_out_of_bounds.any():
        self.alpha.data[mask_out_of_bounds] = torch.abs(torch.sin(self.alpha.data[mask_out_of_bounds]))

    a1 = self.a1
    b1 = self.b1
    c1 = self.c1
    d1 = self.d1
    z = fdeint1(a1, b1, c1, d1, func, state, learnable_t=self.alpha, t=self.opt['time'], fc_layer0=self.fc0, fc_layer1=self.fc1, step_size=self.opt['step_size'], method=self.opt['method'])

    return z

  def __repr__(self):
    return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
           + ")"
