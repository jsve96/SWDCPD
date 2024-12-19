import numpy as np
import torch
from scipy.stats import norm,gamma

#shape rate notation for Gamma distribution
def mom_estimates(X):
    mean_X = np.mean(X)
    var_X = np.var(X, ddof=1)
    alpha_hat = mean_X**2 / var_X
    beta_hat = mean_X/var_X
    return alpha_hat, beta_hat


class TimeseriesDataset(torch.utils.data.Dataset):   
    def __init__(self, X, seq_len=10,split=0.5):
        self.X = X
        #self.y = y
        self.seq_len = seq_len
        self.split = split

    def __len__(self):
        return self.X.__len__() - (self.seq_len-1)

    def __getitem__(self, index):
        window = index + int(self.seq_len*self.split)
        #print(index,window)
        #self.seq_len - int(self.seq_len//2)
        #print(window)
        return (self.X[index:window],self.X[window:window+ int(self.seq_len*(1-self.split))])
    


#def project_and_calc_dist_torch(X, Y, theta, p,device='cpu'):
   # X = X.to(device)
   # Y = Y.to(device)
   # theta = theta.to(device)
   # x_proj = torch.matmul(X, theta.T)
   # y_proj = torch.matmul(Y, theta.T)

   # qs = torch.linspace(0, 1, 100, device=device)

    # Compute quantiles for x_proj and y_proj
   # xp_quantiles = torch.quantile(x_proj, qs, dim=0, interpolation="lower")
   # yp_quantiles = torch.quantile(y_proj, qs, dim=0, interpolation="lower")

    # Calculate the p-distance between quantiles
   # dist_p = torch.abs(xp_quantiles - yp_quantiles)**p

   # return dist_p


def project_and_calc_dist_torch(X, Y, theta, p,device='cpu'):
    X ,Y, theta= X.to(device),Y.to(device), theta.T.to(device)

    x_proj = torch.matmul(X, theta)
    y_proj = torch.matmul(Y, theta)


    qs = torch.linspace(0, 1, 100, device=device)

    # Compute quantiles for x_proj and y_proj
    xp_quantiles = torch.quantile(x_proj, qs, dim=0, interpolation="linear")
    yp_quantiles = torch.quantile(y_proj, qs, dim=0, interpolation="linear")

    # Calculate the p-distance between quantiles
    dist_p = torch.pow(torch.abs(xp_quantiles - yp_quantiles),p)

    return dist_p


def sample_theta_torch(X, num_samples=10, device='cpu'):
    """
    Samples random normalized vectors (theta) using PyTorch.

    Args:
        X (torch.Tensor): Input tensor of shape (n_samples, d).
        num_samples (int): Number of random samples to generate.
        device (str): Device to perform computations on ('cpu' or 'cuda').

    Returns:
        torch.Tensor: Normalized random vectors of shape (num_samples, d).
    """
    _, d = X.shape
    # Generate random values
    theta = torch.randn(num_samples, d, device=device)
    # Compute the norm along the last dimension
    theta_norm = torch.norm(theta, dim=1, keepdim=True)
    # Normalize each vector
    theta_normed = theta / theta_norm
    return theta_normed


def gamma_conf_interval(step, a, b, confidence=0.95):
    alpha = step * a
    lower_bound = gamma.ppf((1 - confidence) / 2, alpha, scale=b)
    upper_bound = gamma.ppf(1 - (1 - confidence) / 2, alpha, scale=b)
    return lower_bound, upper_bound
