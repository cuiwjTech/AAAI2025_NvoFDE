import torch
from torch import nn
import torch.nn.functional as F
from base_classes import BaseGNN
from model_configurations import set_block, set_function


# Define the GNN model. 2024.8.10 version
class GNN(BaseGNN):
  def __init__(self, opt, dataset, device=torch.device('cpu')):
    super(GNN, self).__init__(opt, dataset, device)
    self.f = set_function(opt)
    block = set_block(opt)
    time_tensor = torch.tensor([0, self.T]).to(device)
    self.odeblock = block(self.f, opt, dataset.data, device, t=time_tensor).to(device)

  def forward(self, x):
    if self.opt['use_labels']:
      y = x[:, -self.num_classes:]
      x = x[:, :-self.num_classes]


    x = F.dropout(x, self.opt['input_dropout'], training=self.training)
    x = self.m1(x)
    if self.opt['use_mlp']:
      x = F.dropout(x, self.opt['dropout'], training=self.training)
      x = F.dropout(x + self.m11(F.relu(x)), self.opt['dropout'], training=self.training)
      x = F.dropout(x + self.m12(F.relu(x)), self.opt['dropout'], training=self.training)
    # todo investigate if some input non-linearity solves the problem with smooth deformations identified in the ANODE paper

    if self.opt['use_labels']:
      x = torch.cat([x, y], dim=-1)

    if self.opt['batch_norm']:
      x = self.bn_in(x)

    # Solve the initial value problem of the ODE.
    x_init = x.clone()

    if 'graphcon' in self.opt['function']:
      x = torch.cat([x, x_init], dim=-1)
      self.odeblock.set_x0(x)
      z = self.odeblock(x)
      z = z[:,self.opt['hidden_dim']:]
    else:

      self.odeblock.set_x0(x)
      z = self.odeblock(x)



    z = F.relu(z)

    if self.opt['fc_out']:
      z = self.fc(z)
      z = F.relu(z)


    z = F.dropout(z, self.opt['dropout'], training=self.training)


    z = self.m2(z)
    return z
  




