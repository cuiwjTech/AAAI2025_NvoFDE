import torch
import math
from torch import nn
from function_transformer_attention import SpGraphTransAttentionLayer
from base_classes import ODEblock
import os
import numpy as np
import random
from utils import get_rw_adj
from torchfde import fdeint, fdeint1, fdeint2


class AttODEblock_FRAC5(ODEblock):
  def __init__(self, odefunc,  opt, data,  device, t=torch.tensor([0, 1])):
    super(AttODEblock_FRAC5, self).__init__(odefunc,  opt, data, device, t)
    
    self.odefunc = odefunc( opt['hidden_dim'], opt['hidden_dim'], opt, data, device)
    edge_index, edge_weight = get_rw_adj(data.edge_index, edge_weight=data.edge_attr, norm_dim=1,
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
    self.multihead_att_layer = SpGraphTransAttentionLayer(opt['hidden_dim'], opt['hidden_dim'], opt,device, edge_weights=self.odefunc.edge_weight).to(device)
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

  
  def get_attention_weights(self, x):
    attention, values = self.multihead_att_layer(x, self.odefunc.edge_index)
    return attention

  def forward(self, x):


    t = self.t.type_as(x)

    self.odefunc.attention_weights = self.get_attention_weights(x)


    func = self.odefunc
    state = x

    a1 = self.a1
    b1 = self.b1
    c1 = self.c1
    d1 = self.d1


    mask_out_of_bounds = (self.alpha.data <= 0) | (self.alpha.data > 1)

    if mask_out_of_bounds.any():
        self.alpha.data[mask_out_of_bounds] = torch.abs(torch.sin(self.alpha.data[mask_out_of_bounds]))



    z = fdeint1(a1, b1, c1, d1, func, state, learnable_t=self.alpha, t=self.opt['time'], fc_layer0=self.fc0,fc_layer1=self.fc1, step_size=self.opt['step_size'], method=self.opt['method'])


    return z


  def __repr__(self):
    return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
           + ")"

