import torch
import numpy as np
import pandas as pd
from ChangeDetection.utilsCPD import *
from scipy.stats import norm,gamma
from tqdm import tqdm
from typing import List
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



def gamma_conf_interval(step,a, b, confidence=0.95):
    alpha = step * a
    #lower_bound = gamma.ppf((1 - confidence), alpha, scale=b)
    upper_bound = gamma.ppf(1 - (1 - confidence), alpha, scale=b)
    return upper_bound

def CI_Calibration(max_history:int,alphas: List,betas:List,trend:float,significance: float =0.05):
    lookback = min(max_history,len(alphas))
    #interval = np.arange(1,lookback+1,1)
    #w = (1/np.sqrt(interval)**0)/np.sum(1/np.sqrt(interval)**0)
    a = np.mean(alphas[-lookback:])
    #a = np.average(alphas[-lookback:],weights=w)
    b = np.mean(betas[-lookback:])
    #b = np.average(betas[-lookback:],weights=w)
    return trend + gamma_conf_interval(1,a,1/b,1-significance)



class BaseDetector:
    def __init__(self, data, window_length,max_history:int  = 20, significance=0.05, use_cuda: bool = True):
        self.data = data
        self.window_length = window_length
        self.significance = significance
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if use_cuda else 'cpu'
        self.max_lookback = max_history
        self.loss_segments = []
        self.alphas = []
        self.betas = []
        self.trend = []
        self.cusum = 0
        self.upper = []
        self.first = True
        self.change_points = {'loc': [], 'value': []}
        self.cumsum = []

    def process_dataloader(self, n_theta:int = 500,p:int = 2,split: float = 0.5):
        theta = sample_theta_torch(self.data,n_theta,device=self.device)
        dataloader = DataLoader(TimeseriesDataset(self.data.to_numpy(),self.window_length,split=split))

        for i, d in enumerate(tqdm(dataloader)):
            x_ref, x_cur = d[0].squeeze(0).to(self.device), d[1].squeeze(0).to(self.device)
            loss = project_and_calc_dist_torch(x_ref, x_cur, theta, p=p).mean(axis=0).detach().cpu().numpy()
            self.loss_segments.append(loss)
            self.cusum += loss.mean()
            self.cumsum.append(self.cusum)

            if i > 0:
                if self.cusum >= self.upper[-1]:
                    if self.first:
                        print(f"Change detected at: {i + self.window_length} \nInitiate new segment")
                        self.change_points['loc'].append(i + self.window_length)
                        self.change_points['value'].append(self.upper[-1] - self.cusum)
                        self.first = False
                else:
                    if not self.first:
                        self.first = True

            self.trend.append(self.cusum)
            a_hat, b_hat = mom_estimates(loss)
            self.alphas.append(a_hat)
            self.betas.append(b_hat)
            self.upper.append(CI_Calibration(self.max_lookback,self.alphas, self.betas, self.trend[-1]))

    def evaluate(self, ground_truth: List[int],tolerance: int):
        f1 = f_measure({1: ground_truth}, self.change_points['loc'], tolerance)
        coverage = covering({0: ground_truth}, self.change_points['loc'], self.data.shape[0])
        print(f"F1 score: {f1}")
        print(f"Covering: {coverage}")

    def plot(self,ground_truth):
        fig, ax = plt.subplots()
        t = np.arange(self.window_length+1,len(self.cumsum)+self.window_length,1)
        ax.plot(t,np.subtract(self.upper[:-1],self.cumsum[1:]),label='Distance to Upper bound',color='black',alpha=1,lw=0.5)
        ax.plot(np.array(self.change_points['loc']),self.change_points['value'],'o',color='red',ms=4,label='Change Points')
        ax.plot(ground_truth,np.zeros(len(ground_truth)),'o',color='blue',ms=4,label='True Change Points')
        ax.set_xlabel('Time')
        fig.legend()
        return fig,ax