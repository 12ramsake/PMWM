# coding: utf-8
import torch
import argparse
import numpy as np
import math
import scipy.stats
import matplotlib.pyplot as plt
import diffprivlib as dpl


'''
extra functions
'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_budget', default=.5, type=float, help='Total privacy budget')
    parser.add_argument('--d', default=10, type=int, help='Feature dimension (dimension of synthetic data)')
    parser.add_argument('--n', default=3000, type=int, help='Number of samples to synthesize (for synthetic data)')
    parser.add_argument('--u', default=33, type=float, help='Initial upper bound for covariance')
    
    parser.add_argument('--fig_title', default=None, type=str, help='figure title')
    parser.add_argument('-f', default=None, type=str, help='needed for ipython starting')
    
    opt = parser.parse_args()
    return opt

def cov_nocenter(X):
    cov = torch.mm(X.t(), X)/X.size(0)
    return cov

def cov(X):
    X = X - X.mean(0)
    cov = torch.mm(X.t(), X)/X.size(0)
    return cov

'''
PSD projection
'''
def psd_proj_symm(S):
    U, D, V_t = torch.svd(S)
    D = torch.clamp(D, min=0, max=None).diag()
    A = torch.mm(torch.mm(U, D), U.t()) 
    return A

'''
Mean Estimation Methods --------------------------------------------------------
'''

'''
Fine mean estimation algorithm 
 - list params are purely for graphing purposes and can be ignored if not needed
returns: fine DP estimate for mean
'''
def fineMeanEst(x, sigma, R, epsilon, epsilons=[], sensList=[], rounding_outliers=False):
    B = R+sigma*3
    sens = 2*B/(len(x)*epsilon) 
    epsilons.append([epsilon])
    sensList.append([sens])
    if rounding_outliers:
        for i in x:
            if i > B:
                i = B
            elif i < -1*B:
                i =  -1*B
    noise = np.random.laplace(loc=0.0, scale=sens)
    result = sum(x)/len(x) + noise 
    return result

'''
Coarse mean estimation algorithm with Private Histogram
returns: [start of intrvl, end of intrvl, freq or probability], bin number
- the coarse mean estimation would just be the midpoint of the intrvl (in case this is needed)
'''
def privateRangeEst(x, epsilon, delta, alpha, R, sd):
    # note alpha ∈ (0, 1/2)
    r = int(math.ceil(R/sd))
    bins = {}
    for i in range(-1*r,r+1):
        start = (i - 0.5)*sd # each bin is s ((j − 0.5)σ,(j + 0.5)σ]
        end = (i + 0.5)*sd 
        bins[i] = [start, end, 0] # first 2 elements specify intrvl, third element is freq
    # note: epsilon, delta ∈ (0, 1/n) based on https://arxiv.org/pdf/1711.03908.pdf Lemma 2.3
    # note n = len(x)
    # set delta here
    L = privateHistLearner(x, bins, epsilon, delta, r, sd)
    return bins[L], L


# helper function
# returns: max probability bin number
def privateHistLearner(x, bins, epsilon, delta, r, sd): # r, sd added to transmit info
    # fill bins
    max_prob = 0
    max_r = 0

    # creating probability bins
    for i in x:
        r_temp = int(round(i/sd))
        if r_temp in bins:
            bins[r_temp][2] += 1/len(x)
        
    for r_temp in bins:
        noise = np.random.laplace(loc=0.0, scale=2/(epsilon*len(x)))
        if delta == 0 or r_temp < 2/delta:
            # epsilon DP case
            bins[r_temp][2] += noise
        else:
            # epsilon-delta DP case
            if bins[r_temp][2] > 0:
                bins[r_temp][2] += noise
                t = 2*math.log(2/delta)/(epsilon*len(x)) + (1/len(x))
                if bins[r_temp][2] < t:
                    bins[r_temp][2] = 0
        
        if bins[r_temp][2] > max_prob:
            max_prob = bins[r_temp][2]
            max_r = r_temp
    return max_r


'''
Two shot algorithm
- may want to optimize distribution ratio between fine & coarse estimation

eps1 = epsilon for private histogram algorithm
eps2 = epsilon for fine mean estimation algorithm

returns: DP estimate for mean
'''
def twoShot(x, eps1, eps2, delta, R, sd):
    alpha = 0.5
    # coarse estimation
    [start, end, prob], r = privateRangeEst(x, eps1, delta, alpha, R, sd)
    for i in range(len(x)):
        if x[i] < start - 3*sd:
            x[i] = start - 3*sd
        elif x[i] > end + 3*sd:
            x[i] = end + 3*sd
    # fine estimation with smaller range (less sensitivity)
    est = fineMeanEst(x, sd, 3.5*sd, eps2)
    return est


'''
Privately estimating covariance.
'''




def cov_est_step_1D(X, A, rho, cur_iter, args,beta):
    """
    One step of multivariate covariance estimation, scale cov a.
    """
    # print(X.shape)
    if len(X.shape)>2:
        # print(X.shape)
        X=torch.squeeze(X)
        # print('after' ,X.shape)
    n, d = X.shape

    #Hyperparameters
    gamma = gaussian_tailbound(d, beta)
    # Was here before
    # gamma = gaussian_tailbound(d, 0.1)
    # eta = 0.5*(2*(np.sqrt(d/n)) + (np.sqrt(d/n))**2)
    eta = 2*(np.sqrt(d/n) + np.sqrt(2*np.log(2/beta)/n)) + (np.sqrt(d/n) + np.sqrt(2*np.log(2/beta)/n))**2
    # We widen nu since its infinite at d=1
    dd=2
    nu = (gamma**2 / (n*np.sqrt(rho))) * (2*np.sqrt(dd) + 2*dd**(1/16) * np.log(dd)**(1/3) + (6*(1 + (np.log(dd)/dd)**(1/3))*np.sqrt(np.log(dd)))/(np.sqrt(np.log(1 + (np.log(dd)/dd)**(1/3)))) + 2*np.sqrt(2*np.log(1/beta)))

    
    #truncate points
    # W = torch.mm(X, A)
    W = torch.mm(X, A.t())
    W_norm = np.sqrt((W**2).sum(-1, keepdim=True))
    norm_ratio = gamma / W_norm
    large_norm_mask = (norm_ratio < 1).squeeze()
    
    W[large_norm_mask] = W[large_norm_mask] * norm_ratio[large_norm_mask]
    
    # noise
    Y = torch.randn(d, d)
    noise_var = (gamma**4/(rho*n**2))
    Y *= np.sqrt(noise_var)    
    #can also do Y = torch.triu(Y, diagonal=1) + torch.triu(Y).t()
    Y = torch.triu(Y)
    Y = Y + Y.t() - Y.diagonal().diag_embed() # Don't duplicate diagonal entries
    Z = (torch.mm(W.t(), W))/n
    #add noise    
    Z = Z + Y
    #ensure psd of Z
    if Z<0:
        Z=torch.zeros((1,1))+0.0001
    # Z = psd_proj_symm(Z)
    
    U = Z + (nu+eta)*torch.eye(d)
    inv = torch.inverse(U)
    inv_sqrt=inv**(1/2)
    # inv_sqrt=compute_sqrt_mat(inv)
    # invU, invD, invV = inv.svd()
    # inv_sqrt = torch.mm(invU, torch.mm(invD.sqrt().diag_embed(), invV.t()))
    A = torch.mm(inv_sqrt, A)
    return A, Z



# def compute_sqrt_mat(A):
#     U, D, V = A.svd()
#     inv_sqrt = torch.mm(U, torch.mm(D.sqrt().diag_embed(), V.t()))
#     return inv_sqrt


def cov_est_1D(X, args,beta=0.1):
    """
    Multivariate covariance estimation.
    Returns: zCDP estimate of cov.
    """
    A = torch.eye(args.d) / np.sqrt(args.u)
    assert len(args.rho) == args.t
    
    for i in range(args.t-1):
        A, Z = cov_est_step_1D(X, A, args.rho[i], i, args,beta/(4*(args.t-1)))
    A_t, Z_t = cov_est_step_1D(X, A, args.rho[-1], args.t-1, args,beta/4)
    # A.inverse()
    # print(A)
    cov = torch.mm(torch.mm(A.inverse(), Z_t), A.inverse().t())
    return cov






def gaussian_tailbound(d,b):
    return ( d + 2*( d * np.log(1/b) )**0.5 + 2*np.log(1/b) )**0.5

def mahalanobis_dist(M, Sigma):
    Sigma_inv = torch.inverse(Sigma)
    U_inv, D_inv, V_inv = Sigma_inv.svd()
    Sigma_inv_sqrt = torch.mm(U_inv, torch.mm(D_inv.sqrt().diag_embed(), V_inv.t()))
    M_normalized = torch.mm(Sigma_inv_sqrt, torch.mm(M, Sigma_inv_sqrt))
    return torch.norm(M_normalized - torch.eye(M.size()[0]), 'fro')

''' 
Functions for mean estimation
'''

##    X = dataset
##    c,r = prior knowledge that mean is in B2(c,r)
##    t = number of iterations
##    Ps = 
def multivariate_mean_iterative(X, c, r, t, Ps,beta=0.1):
    for i in range(t-2):
        c, r = multivariate_mean_step(X, c, r, Ps[i],beta/(4*(t-1)))
    c, r = multivariate_mean_step(X, c, r, Ps[t-1],beta/4)
    return c

def multivariate_mean_step(X, c, r, p,beta):
    n, d = X.shape

    ## Determine a good clipping threshold
    gamma = gaussian_tailbound(d,beta)
    clip_thresh = min((r**2 + 2*r*3 + gamma**2)**0.5 , r + gamma) #3 in place of sqrt(log(2/beta))
        
    ## Round each of X1,...,Xn to the nearest point in the ball B2(c,clip_thresh)
    x = X - c
    mag_x = torch.linalg.norm(x, axis=1)
    outside_ball = (mag_x > clip_thresh)
    x_hat = (x.T / mag_x).T
    if torch.sum(outside_ball)>0:
        X[outside_ball] = c.float() + (x_hat[outside_ball].float() * clip_thresh)
    
    ## Compute sensitivity
    delta = 2*clip_thresh/float(n)
    # print(delta)
    # print(p)
    sd = delta/(2*p)**0.5
    
    ## Add noise calibrated to sensitivity
    Y = np.random.normal(0, sd, size=d)
    c = torch.sum(X, axis=0)/float(n) + Y
    r = ( 1/float(n) + sd**2 )**0.5 * gaussian_tailbound(d,0.01)
    return c, r

def L1(est): # assuming 0 vector is gt
    return np.sum(np.abs(est))
    
def L2(est): # assuming 0 vector is gt
    return np.linalg.norm(est)

def overall_mean_1D(X,args):
    # n, d = X.shape
    # print('before', X.shape)
    if len(X.shape)>2:
        # print(X.shape)
        X=torch.squeeze(X)
    if len(X.shape)<2:
        X=X.view(X.shape[0], 1)
    row_diff = torch.diff(X, axis=0)
    
    Sigma=cov_est_1D(row_diff,args)/2

    # U, S, Vh =torch.linalg.svd(Sigma)
    # D = torch.diag(torch.sqrt(S))
    # sqrt_mat = U @ D @ Vh
    sqrt_mat = Sigma**(1/2)
    adj = torch.linalg.inv(sqrt_mat)

    # print('after', X.shape)
    whitened=X @ adj
    # args.t=3
    # args.r=10*np.sqrt(args.d)
    # args.Ps= torch.tensor([1/3.0, 1/2.0, 1.0])
    # multivariate_mean_iterative(X, c, r, t, Ps)
    mean_est = multivariate_mean_iterative(whitened,args.c,args.r,args.t,args.Ps)
    mean_est = mean_est.float() @ sqrt_mat
    return [mean_est, Sigma]



def COINPRESS1D(X,n,d,rho,c,r,u):
    args = parse_args()
    args.d = d
    args.n = n
    nm=torch.tensor([1/3.0, 1/2.0, 1.0])
    args.rho= rho*(nm/nm.sum())/2
    args.t=3
    args.c=c
    args.r=r
    args.u=u
    # args.Ps= [1.0/3.0, 1.0/2.0, 1.0]
    args.Ps= rho*(nm/nm.sum())/2
    return overall_mean_1D(X,args)



