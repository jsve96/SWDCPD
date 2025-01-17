import numpy as np
import torch
from scipy.stats import norm,gamma
import pandas as pd

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




def load_master_data(data_path="/datasets/has2023_master.csv.zip"):
    """
    Load the given CSV file containing the labelled challenge data.
    Returns a pandas DataFrame where each column is a sensor measurement
    or label and each row corresponds to a single time series.

    Parameters
    ----------
    data_path : str, default: "../datasets/has2023_master.csv.zip".
        Path to the csv file to be loaded.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the sensor data for the challenge.

    Examples
    --------
    >>> data = load_master_data()
    >>> data.head()
    """
    np_cols = ["change_points", "activities", "x-acc", "y-acc", "z-acc",
               "x-gyro", "y-gyro", "z-gyro",
               "x-mag", "y-mag", "z-mag",
               "lat", "lon", "speed"]
    converters = {
        col: lambda val: np.array([]) if len(val) == 0 else np.array(eval(val)) for col
        in np_cols}
    return pd.read_csv(data_path, converters=converters, compression="zip")


def load_data(data_path="../datasets/has2023.csv.zip"):
    """
    Load the given CSV file containing the sensor data for the challenge.
    Returns a pandas DataFrame where each column is a sensor measurement and
    each row corresponds to a single time series of sensor data.

    Parameters
    ----------
    data_path : str, default: "../datasets/has2023.csv.zip".
        Path to the csv file to be loaded.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the sensor data for the challenge.

    Examples
    --------
    >>> data = load_data()
    >>> data.head()
    """
    np_cols = ["x-acc", "y-acc", "z-acc",
               "x-gyro", "y-gyro", "z-gyro",
               "x-mag", "y-mag", "z-mag",
               "lat", "lon", "speed"]
    converters = {
        col: lambda val: np.array([]) if len(val) == 0 else np.array(eval(val)) for col
        in np_cols}
    return pd.read_csv(data_path, converters=converters, compression="zip")


### from https://github.com/alan-turing-institute/TCPDBench/blob/master/analysis/scripts/metrics.py
def true_positives(T, X, margin=5):
    """Compute true positives without double counting

    >>> true_positives({1, 10, 20, 23}, {3, 8, 20})
    {1, 10, 20}
    >>> true_positives({1, 10, 20, 23}, {1, 3, 8, 20})
    {1, 10, 20}
    >>> true_positives({1, 10, 20, 23}, {1, 3, 5, 8, 20})
    {1, 10, 20}
    >>> true_positives(set(), {1, 2, 3})
    set()
    >>> true_positives({1, 2, 3}, set())
    set()
    """
    # make a copy so we don't affect the caller
    X = set(list(X))
    TP = set()
    for tau in T:
        close = [(abs(tau - x), x) for x in X if abs(tau - x) <= margin]
        close.sort()
        if not close:
            continue
        dist, xstar = close[0]
        TP.add(tau)
        X.remove(xstar)
    return TP


def f_measure(annotations, predictions, margin=5, alpha=0.5, return_PR=False):
    """Compute the F-measure based on human annotations.

    annotations : dict from user_id to iterable of CP locations
    predictions : iterable of predicted CP locations
    alpha : value for the F-measure, alpha=0.5 gives the F1-measure
    return_PR : whether to return precision and recall too

    Remember that all CP locations are 0-based!

    >>> f_measure({1: [10, 20], 2: [11, 20], 3: [10], 4: [0, 5]}, [10, 20])
    1.0
    >>> f_measure({1: [], 2: [10], 3: [50]}, [10])
    0.9090909090909091
    >>> f_measure({1: [], 2: [10], 3: [50]}, [])
    0.8
    """
    # ensure 0 is in all the sets
    Tks = {k + 1: set(annotations[uid]) for k, uid in enumerate(annotations)}
    for Tk in Tks.values():
        Tk.add(0)

    X = set(predictions)
    X.add(0)

    Tstar = set()
    for Tk in Tks.values():
        for tau in Tk:
            Tstar.add(tau)

    K = len(Tks)

    P = len(true_positives(Tstar, X, margin=margin)) / len(X)

    TPk = {k: true_positives(Tks[k], X, margin=margin) for k in Tks}
    R = 1 / K * sum(len(TPk[k]) / len(Tks[k]) for k in Tks)

    F = P * R / (alpha * R + (1 - alpha) * P)
    if return_PR:
        return F, P, R
    return F


def overlap(A, B):
    """Return the overlap (i.e. Jaccard index) of two sets

    >>> overlap({1, 2, 3}, set())
    0.0
    >>> overlap({1, 2, 3}, {2, 5})
    0.25
    >>> overlap(set(), {1, 2, 3})
    0.0
    >>> overlap({1, 2, 3}, {1, 2, 3})
    1.0
    """
    return len(A.intersection(B)) / len(A.union(B))


def partition_from_cps(locations, n_obs):
    """Return a list of sets that give a partition of the set [0, T-1], as
    defined by the change point locations.

    >>> partition_from_cps([], 5)
    [{0, 1, 2, 3, 4}]
    >>> partition_from_cps([3, 5], 8)
    [{0, 1, 2}, {3, 4}, {5, 6, 7}]
    >>> partition_from_cps([1,2,7], 8)
    [{0}, {1}, {2, 3, 4, 5, 6}, {7}]
    >>> partition_from_cps([0, 4], 6)
    [{0, 1, 2, 3}, {4, 5}]
    """
    T = n_obs
    partition = []
    current = set()

    all_cps = iter(sorted(set(locations)))
    cp = next(all_cps, None)
    for i in range(T):
        if i == cp:
            if current:
                partition.append(current)
            current = set()
            cp = next(all_cps, None)
        current.add(i)
    partition.append(current)
    return partition


def cover_single(S, Sprime):
    """Compute the covering of a segmentation S by a segmentation Sprime.

    This follows equation (8) in Arbaleaz, 2010.

    >>> cover_single([{1, 2, 3}, {4, 5, 6}], [{1, 2, 3}, {4, 5}, {6}])
    0.8333333333333334
    >>> cover_single([{1, 2, 3, 4, 5, 6}], [{1, 2, 3, 4}, {5, 6}])
    0.6666666666666666
    >>> cover_single([{1, 2, 3}, {4, 5, 6}], [{1, 2}, {3, 4}, {5, 6}])
    0.6666666666666666
    >>> cover_single([{1}, {2}, {3}, {4, 5, 6}], [{1, 2, 3, 4, 5, 6}])
    0.3333333333333333
    """
    T = sum(map(len, Sprime))
    assert T == sum(map(len, S))
    C = 0
    for R in S:
        C += len(R) * max(overlap(R, Rprime) for Rprime in Sprime)
    C /= T
    return C


def covering(annotations, predictions, n_obs):
    """Compute the average segmentation covering against the human annotations.

    annotations : dict from user_id to iterable of CP locations
    predictions : iterable of predicted Cp locations
    n_obs : number of observations in the series

    >>> covering({1: [10, 20], 2: [10], 3: [0, 5]}, [10, 20], 45)
    0.7962962962962963
    >>> covering({1: [], 2: [10], 3: [40]}, [10], 45)
    0.7954144620811286
    >>> covering({1: [], 2: [10], 3: [40]}, [], 45)
    0.8189300411522634

    """
    Ak = {
        k + 1: partition_from_cps(annotations[uid], n_obs)
        for k, uid in enumerate(annotations)
    }
    pX = partition_from_cps(predictions, n_obs)

    Cs = [cover_single(Ak[k], pX) for k in Ak]
    return sum(Cs) / len(Cs)