import numpy as np
import math
from collections import defaultdict


######## Pure DP Estimates



def unboundedQuantileUpper(data , l, b, q, eps_1 , eps_2 ):
    d = defaultdict ( int )
    for x in data :
        i = math.log ( max([x-l+1,b]),b) // 1
        d[i] += 1
    t = q * len( data ) + np.random.exponential(scale=1/ eps_1 , size=1)
    cur , i = 0, 0
    while True :
        cur += d[i]
        i += 1
        if (cur + np.random.exponential(scale=1/ eps_2 , size=1)) > t:
            break
    try:
        res=b ** i + l - 1
    except:
        print('overflow')
        res=1000000
    return res
    # if res<l:
    #     return res



def unboundedQuantile(data , l,u, b, q, eps_1 , eps_2 ):
    if q<1/2:
        return np.max([-unboundedQuantileUpper(-data , -u, b, 1-q, eps_1 , eps_2 ),l])
    else:
        return np.min([unboundedQuantileUpper(data , l, b, q, eps_1 , eps_2 ),u])


def private_tm(data, a,b, eps_1,eps_2,eps_3,beta=1.01,constant=10,eta=0.01,seed=np.nan):
    if seed is not np.nan:
        np.random.seed(seed=seed)    
    n = len(data)
    cap=max([0.025,eta])
    trim_param=min([max([constant/n,eta]),cap])
    m=int(n/2)
    ub = unboundedQuantile(data[0:m] , a,b , beta, 1-trim_param,eps_1/2,eps_2/2 )
    lb = unboundedQuantile(data[0:m] , a,b , beta, trim_param, eps_1/2,eps_2/2 )
    Z=np.clip(data[m:],lb,ub)
    np_mean=np.mean(Z)
    V=np.random.laplace(scale=1/ eps_3 , size=1)
    return (np_mean+2*V*(ub-lb)/n)[0]



def private_tm_prac(data, a,b, eps_1,eps_2,eps_3,beta=1.01,constant=10,eta=0.01,seed=np.nan):
   if seed is not np.nan:
        np.random.seed(seed=seed)    
   n = len(data)
   cap=max([0.025,eta])
   trim_param=min([max([constant/n,eta]),cap])
   ub = unboundedQuantile(data , a,b , beta, 1-trim_param,eps_1/2,eps_2/2 )
   lb = unboundedQuantile(data , a,b , beta, trim_param, eps_1/2,eps_2/2 )
   Z=np.clip(data,lb,ub)
   np_mean=np.mean(Z)
   V=np.random.laplace(scale=1/ eps_3 , size=1)
   return (np_mean+V*(ub-lb)/n)[0]

# old one 
# def private_tm(data, a,b,eps_1,eps_2,eps_3,beta=1.01,delta=0.1,eta=0.01):
#     n = len(data)
#     # print(data)
#     # trim_param=eta+np.log(1/delta)/n
#     trim_param=min([np.max(np.array([40/n,0.05])),0.1])
#     m=int(n/2)
#     # print(trim_param)
#     ub=unboundedQuantile(data[0:m] , a,b , beta, 1-trim_param, eps_1 , eps_2 )
#     lb=unboundedQuantile(data[0:m] , a,b , beta, trim_param, eps_1 , eps_2 )
#     Z=np.clip(data[m:],lb,ub)
#     np_mean=np.mean(Z)
#     V=np.random.laplace(scale=1/ eps_3 , size=1)
#     return (np_mean+2*V*(ub-lb)/n)[0]

######## zCDP Estimates
######## Quantile setimation

def unboundedQuantileUpper_zCDP(data , l, b, q, rho_1,rho_2 ):
    d = defaultdict ( int )
    for x in data :
        i = math.log ( max([x-l+1,b]),b) // 1
        d[i] += 1
    t = q * len( data ) + np.random.normal(scale=1/ np.sqrt(rho_1) , size=1)
    cur , i = 0, 0
    while True :
        cur += d[i]
        i += 1
        if (cur + np.random.normal(scale=1/ np.sqrt(rho_2) , size=1)) > t:
            break
    try:
        res=b ** i + l - 1
    except:
        print('warning')
        res=1000000
    # print(res)
    return res
    # if res<l:
    #     return res


def unboundedQuantile_zCDP(data , l,u, b, q, rho_1,rho_2 ):
    if q<1/2:
        return np.max([-unboundedQuantileUpper_zCDP(-data , -u, b, 1-q,  rho_1,rho_2 ),l])
    else:
        return np.min([unboundedQuantileUpper_zCDP(data , l, b, q, rho_1,rho_2 ),u])


######## Trimmed mean zCDP

def private_tm_zCDP(data, a,b, rho_1,rho_2,rho_3,beta=1.01,constant=10,eta=0.01,seed=np.nan):
    if seed is not np.nan:
        np.random.seed(seed=seed)    
    n = len(data)
    cap=max([0.025,eta])
    trim_param=min([max([constant/n,eta]),cap])
    m=int(n/2)
    ub = unboundedQuantile_zCDP(data[0:m] , a,b , beta, 1-trim_param, rho_1/2,rho_2/2 )
    lb = unboundedQuantile_zCDP(data[0:m] , a,b , beta, trim_param, rho_1/2,rho_2/2 )
    if(ub<lb):
        # print('warning ! UB<LB??')
        # print(trim_param)
        # print(lb)
        # print(ub)
        lb=a
        ub=b
    Z=np.clip(data[m:],lb,ub)
    np_mean=np.mean(Z)
    V=np.random.normal(scale=1/ np.sqrt(2*rho_3) , size=1)
    return (np_mean+2*V*(ub-lb)/n)[0]

#uses all data
def private_tm_zCDP_mod_1(data, a,b, rho_1,rho_2,rho_3,beta=1.01,constant=10,eta=0.01,seed=np.nan):
    if seed is not np.nan:
        np.random.seed(seed=seed)    
    n = len(data)
    cap=max([0.025,eta])
    trim_param=min([max([constant/n,eta]),cap])
    ub = unboundedQuantile_zCDP(data , a,b , beta, 1-trim_param, rho_1/2,rho_2/2 )
    lb = unboundedQuantile_zCDP(data , a,b , beta, trim_param, rho_1/2,rho_2/2 )
    if(ub<lb):
        # print('warning ! UB<LB??')
        # print(trim_param)
        # print(lb)
        # print(ub)
        lb=a
        ub=b
    Z=np.clip(data,lb,ub)
    np_mean=np.mean(Z)
    V=np.random.normal(scale=1/ np.sqrt(2*rho_3) , size=1)
    # print((ub-lb))
    return (np_mean+V*(ub-lb)/n)[0]





def private_tm_zCDP_split(data, a,b, rho_1,rho_2,rho_3,beta=1.01,constant=10,eta=0.01,seed=np.nan):
    if seed is not np.nan:
        np.random.seed(seed=seed)    
    n = len(data)
    cap=max([0.025,eta])
    trim_param=min([max([constant/n,eta]),cap])
    m=int(n/2)
    ub = unboundedQuantile_zCDP(data , a,b , beta, 1-trim_param, rho_1/2,rho_2/2 )
    lb = unboundedQuantile_zCDP(data , a,b , beta, trim_param, rho_1/2,rho_2/2 )
    if(ub<lb):
        # print('warning ! UB<LB??')
        # print(trim_param)
        # print(lb)
        # print(ub)
        lb=a
        ub=b
    Z=np.clip(data,np.quantile(data,trim_param) ,np.quantile(data,1-trim_param))
    np_mean=np.mean(Z)
    V=np.random.normal(scale=1/ np.sqrt(2*rho_3) , size=1)
    # print((ub-lb))
    return [np_mean,V*(ub-lb)/n]




#uses less for quantile
def private_tm_zCDP_mod_2(data, a,b, rho_1,rho_2,rho_3,beta=1.01,constant=10,eta=0.01,seed=np.nan):
    if seed is not np.nan:
        np.random.seed(seed=seed)    
    n = len(data)
    cap=max([0.025,eta])
    trim_param=min([max([constant/n,eta]),cap])
    m=int(n/4)
    ub = unboundedQuantile_zCDP(data[0:m] , a,b , beta, 1-trim_param, rho_1,rho_2 )
    lb = unboundedQuantile_zCDP(data[0:m] , a,b , beta, trim_param, rho_1,rho_2 )
    Z=np.clip(data[m:],lb,ub)
    np_mean=np.mean(Z)
    V=np.random.normal(scale=1/ np.sqrt(2*rho_3) , size=1)
    return (np_mean+4*V*(ub-lb)/(3*n))[0]


#uses more for quantile
def private_tm_zCDP_mod_3(data, a,b, rho_1,rho_2,rho_3,beta=1.01,constant=10,eta=0.01,seed=np.nan):
    if seed is not np.nan:
        np.random.seed(seed=seed)    
    n = len(data)
    cap=max([0.025,eta])
    trim_param=min([max([constant/n,eta]),cap])
    m=int(3*n/4)
    ub = unboundedQuantile_zCDP(data[0:m] , a,b , beta, 1-trim_param, rho_1,rho_2 )
    lb = unboundedQuantile_zCDP(data[0:m] , a,b , beta, trim_param, rho_1,rho_2 )
    Z=np.clip(data[m:],lb,ub)
    np_mean=np.mean(Z)
    V=np.random.normal(scale=1/ np.sqrt(2*rho_3) , size=1)
    return (np_mean+V*(ub-lb)/n)[0]





######## Multivariate Trimmed mean zCDP

def private_tm_zCDP_multi(data, a,b, rho_1,rho_2,rho_3,beta=1.01,constant=10,eta=0.01,seed=np.nan):
    if seed is not np.nan:
        np.random.seed(seed=seed)
    s = data.shape
    n=s[0]
    d=s[1]
    cap=max([0.1,eta])
    trim_param=min([max([constant/n,eta]),cap])
    m=int(n/2)
    Z=data[m:,:].copy()
    sen=0
    for j in range(d):
        ub = unboundedQuantile_zCDP(data[0:m,j] , a,b , beta, 1-trim_param, rho_1,rho_2 )
        lb = unboundedQuantile_zCDP(data[0:m,j] , a,b , beta, trim_param, rho_1,rho_2 )
        Z=np.clip(data[:,j],lb,ub)
        sen+=((ub-lb)**2)/(n**2) 
    sen=2*np.sqrt(sen)
    np_mean=np.mean(Z,axis=0)
    V=np.random.normal(scale=sen/ np.sqrt(2*rho_3) , size=d)
    return np_mean+V

