import torch
from torch import nn
import math
import torch.nn.functional as F
import time
import os


def fractional_pow(base, exponent):
    eps = 1e-4
    return torch.exp(exponent * torch.log(base + eps))

def Predictor(func,y0,beta,tspan,**options):
    """Use one-step Adams-Bashforth (Euler) method to integrate Caputo equation
        D^beta y(t) = f(t,y)
        Args:
          beta: fractional exponent in the range (0,1)
          f: callable(y,t) returning a numpy array of shape (d,)
             Vector-valued function to define the right hand side of the system
          y0: array of shape (d,) giving the initial state vector y(t==0)
          tspan (array): The sequence of time points for which to solve for y.
            These must be equally spaced, e.g. np.arange(0,10,0.005)
            tspan[0] is the intial time corresponding to the initial state y0.
        Returns:
          y: array, with shape (len(tspan), len(y0))
             With the initial value y0 in the first row
        Raises:
          FODEValueError
        See also:
          K. Diethelm et al. (2004) Detailed error analysis for a fractional Adams
             method
          C. Li and F. Zeng (2012) Finite Difference Methods for Fractional
             Differential Equations
        """
    N = len(tspan)

    h = (tspan[-1] - tspan[0]) / (N - 1)

    gamma_beta = 1 / math.gamma(beta)
    fhistory = []
    device = y0.device
    yn = y0.clone()



    for k in range(N):
        tn = tspan[k]
        f_k = func(tn,yn)
        fhistory.append(f_k)

        # can apply short memory here
        if 'memory' not in options:
            memory = k
        else:
            memory = options['memory']
        memory_k = max(0, k - memory)


        j_vals = torch.arange(0, k + 1, dtype=torch.float32,device=device).unsqueeze(1)
        b_j_k_1 = (fractional_pow(h, beta) / beta) * (fractional_pow(k + 1 - j_vals, beta) - fractional_pow(k - j_vals, beta))
        temp_product = torch.stack([b_j_k_1[i] * fhistory[i] for i in range(memory_k,k + 1)])
        b_all_k = torch.sum(temp_product, dim=0)
        yn = y0 + gamma_beta * b_all_k

    del fhistory
    del b_j_k_1
    del temp_product
    return yn







def Predictor5(a1, b1, c1, d1, func, y0, fc_layer0, fc_layer1, learnable_t, tspan, **options):    
    N = len(tspan)
    h = (tspan[-1] - tspan[0]) / (N - 1)
    fhistory = []
    device = y0.device
    yn = y0.clone()
    beta_n_history = []
    calculated_values = []  

    for k in range(N):
        tn = tspan[k]
        f_k = func(tn, yn)
        fhistory.append(f_k)

        beta_n = learnable_t[k]

        # beta_n = periodic_sin_func(beta_n, tspan[-1])
        # beta_n = sigmoid_sin_cos_func(c1, d1, beta_n, tspan[-1])
        
        # beta_n_history.append(beta_n.item())
        gamma_beta_n = 1 / torch.exp(torch.lgamma(beta_n))

        # can apply short memory here
        if 'memory' not in options:
            memory = k
        else:
            memory = options['memory']
        memory_k = max(0, k - memory)


        j_vals = torch.arange(0, k + 1, dtype=torch.float32,device=device).unsqueeze(1)
        b_j_k_1 = (fractional_pow(h, beta_n) / beta_n) * (fractional_pow(k + 1 - j_vals, beta_n) - fractional_pow(k - j_vals, beta_n))
        temp_product = torch.stack([b_j_k_1[i] * fhistory[i] for i in range(memory_k,k + 1)])
        b_all_k = torch.sum(temp_product, dim=0)
        yn = y0 + gamma_beta_n * b_all_k


    del fhistory
    del b_j_k_1
    del temp_product
    del a1
    del b1
    del c1
    del d1

    return yn




def get_timestep_embedding(timesteps, embedding_dim: int):

    timesteps = torch.tensor([timesteps.item()])

    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))

    return emb



def calculate_alpha(t, X, fc_layer0, fc_layer1):

    t = torch.tensor([[t]], dtype=torch.float32).to(X.device)
    pooled_features = torch.max(X, dim=0, keepdim=True)[0]
    pooled_features = F.normalize(fc_layer1(pooled_features), p=2, dim=-1)


    features_dim = pooled_features.shape[-1]
    time_embedding_at_t = get_timestep_embedding(t, features_dim).to(X.device)


    combined_features = pooled_features + time_embedding_at_t
    # combined_features = 0.05 * pooled_features + time_embedding_at_t



    reduced_features = fc_layer0(combined_features).to(X.device)

    reduced_features = reduced_features.squeeze()

    alpha = torch.sigmoid(reduced_features)
    return alpha



def Predictor5_Modify(a1, b1, c1, d1, func, y0, fc_layer0 ,fc_layer1, learnable_t, tspan, **options): 
    N = len(tspan)

    h = (tspan[-1] - tspan[0]) / (N - 1)


    fhistory = []
    device = y0.device
    yn = y0.clone()

    beta_n_history = []
    calculated_values = []  

    for k in range(N):
        tn = tspan[k]
        f_k = func(tn, yn)
        fhistory.append(f_k)




        beta_n = calculate_alpha(tn, yn, fc_layer0, fc_layer1)

        
        beta_n_history.append(beta_n.item())
        gamma_beta_n = 1 / torch.exp(torch.lgamma(beta_n))


        # can apply short memory here
        if 'memory' not in options:

            memory = k
        else:

            memory = options['memory']

        memory_k = max(0, k - memory)



        j_vals = torch.arange(0, k + 1, dtype=torch.float32,device=device).unsqueeze(1)
        b_j_k_1 = (fractional_pow(h, beta_n) / beta_n) * (fractional_pow(k + 1 - j_vals, beta_n) - fractional_pow(k - j_vals, beta_n))
        temp_product = torch.stack([b_j_k_1[i] * fhistory[i] for i in range(memory_k,k + 1)])

        b_all_k = torch.sum(temp_product, dim=0)
        yn = y0 + gamma_beta_n * b_all_k


    del fhistory
    del b_j_k_1
    del temp_product
    del a1
    del b1
    del c1
    del d1

    return yn
























    









