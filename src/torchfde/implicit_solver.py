import torch
import math
import torch.nn.functional as F

def Implicit_l1(func,y0,beta,tspan,**options):
    """Use one-step Implicit_l1 method to integrate Caputo equation
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
    h = (tspan[N - 1] - tspan[0]) / (N - 1)
    yn = y0.clone()
    device = y0.device

    yn_all = []
    u_h = (torch.pow(h, beta) * math.gamma(2 - beta))
    yn_all.append(yn)

    for k in range(1, N):
        tn = tspan[k]
        fhistory_k = func(tn, yn)
        y_sum = 0
        for j in range(0, k - 2):
            R_k_j = torch.pow(k - j, 1 - beta) - torch.pow(k - j - 1, 1 - beta)
            y_sum = y_sum + R_k_j * (yn_all[j + 1] - yn_all[j])
        yn = yn + u_h * fhistory_k - y_sum
        yn_all.append(yn)

    return yn



def get_timestep_embedding(timesteps, embedding_dim: int, device='cpu'):


    timesteps = torch.tensor([timesteps.item()], device=device)


    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)

    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -emb)

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
    time_embedding_at_t = get_timestep_embedding(t, features_dim, X.device).to(X.device)


    combined_features = pooled_features + time_embedding_at_t


    reduced_features = fc_layer0(combined_features).to(X.device)
    reduced_features = reduced_features.squeeze()


    alpha = torch.sigmoid(reduced_features)
   
    return alpha

def fractional_pow(base, exponent):
    eps = 1e-4
    base = torch.tensor(base, dtype=torch.float64)
    exponent = torch.tensor(exponent, dtype=torch.float64)
    return torch.exp(exponent * torch.log(base + eps))







def Implicit_l11(func, y0, fc_layer0, fc_layer1, tspan, **options):


    N = len(tspan)
    h = (tspan[N - 1] - tspan[0]) / (N - 1)
    device = y0.device
    yn = y0.clone().to(device)
    
    yn_all = []
    yn_all.append(yn)
    beta_history = []

    for k in range(1, N):
        tn = tspan[k]
        beta = calculate_alpha(tn, yn, fc_layer0, fc_layer1).to(device)  # 计算变阶beta

        beta_history.append(beta.item())

        u_h = (fractional_pow(h, beta) * torch.tensor(math.gamma(2 - beta), dtype=torch.float64, device=device))


        fhistory_k = func(tn, yn).to(device)


        y_sum = 0


        for j in range(0, k - 2):

            R_k_j = fractional_pow(k - j, 1 - beta) - fractional_pow(k - j - 1, 1 - beta)
            y_sum = y_sum + R_k_j * (yn_all[j + 1] - yn_all[j])


        yn = yn + u_h * fhistory_k - y_sum
        yn_all.append(yn)

    return yn




