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



def estimate_parameter_CI(X,p):
    alpha_hat, beta_hat = mom_estimates(X)
    n= len(X)
    var_alpha_hat = (6* alpha_hat**2) / n  #+ beta_hat**4*var_X
    var_beta_hat = beta_hat**2/(n*alpha_hat) + beta_hat**6/alpha_hat**2*(2*alpha_hat**2)/beta_hat**4/n

    # Confidence level (e.g., 95%)
    confidence_level = p
    z_alpha_2 = norm.ppf(1 - (1 - confidence_level) / 2)  # Critical value for normal distribution

    # Confidence intervals
    alpha_CI = (alpha_hat - z_alpha_2 * np.sqrt(var_alpha_hat), alpha_hat + z_alpha_2 * np.sqrt(var_alpha_hat))
    beta_CI = (beta_hat - z_alpha_2 * np.sqrt(var_beta_hat), beta_hat + z_alpha_2 * np.sqrt(var_beta_hat))

    return alpha_CI,beta_CI

def bootstrap_mom_estimates(X, n_bootstrap=1000, random_seed=None):
    
    if random_seed is not None:
        np.random.seed(random_seed)

    n = len(X)
    alpha_bootstrap = np.empty(n_bootstrap)
    beta_bootstrap = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        # Resample with replacement
        bootstrap_sample = np.random.choice(X, size=n, replace=True)
        # Compute method of moments estimates for the bootstrap sample
        alpha_hat, beta_hat = mom_estimates(bootstrap_sample)
        alpha_bootstrap[i] = alpha_hat
        beta_bootstrap[i] = beta_hat
    
    alpha_ci = np.percentile(alpha_bootstrap, [2.5, 97.5])  # 95% CI for alpha
    beta_ci = np.percentile(beta_bootstrap, [2.5, 97.5])

    return alpha_ci[0], beta_ci[1]



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


