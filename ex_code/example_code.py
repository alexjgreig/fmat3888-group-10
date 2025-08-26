# Black-Scholes functions
import numpy as np
from scipy.stats import norm

def BS_Call_E(S, K, T, r, sig):
    d1 = (np.log(S/K) + (r + 0.5 * sig**2)*T) / (sig*np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T) 
    return (S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)) 

def BS_Put_E(S, K, T, r, sigma): 
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2)*T) / (sigma*np.sqrt(T)) 
    d2 = d1 - sigma * np.sqrt(T) 
    return (-S * norm.cdf(-d1) + K * np.exp(-r*T) * norm.cdf(-d2)) 


def BS_Vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def BS_Implied_Vol(target_value, S, K, T, r ):  #, *args 
    MAX_ITERATIONS = 200
    PRECISION = 1.0e-5
    sigma = 0.5
    for i in range(0, MAX_ITERATIONS):
        price = BS_Call_E(S, K, T, r, sigma)
        vega = BS_Vega(S, K, T, r, sigma)
        diff = target_value - price  # our root
        if (abs(diff) < PRECISION):
            return sigma
        sigma = sigma + diff/vega # f(x) / f'(x)
    return sigma # value wasn't found, return best guess so far


def BS_Vomma(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2)*T) / (sigma*np.sqrt(T)) 
    d2 = d1 - sigma * np.sqrt(T)  

    return BS_Vega(S, K, T, r, sigma) * (d1 * d2 / sigma)

def BS_Barrier_CDI(S,K,Barrier,T,r,sigma):
    
    barr_lambda = ((r) + (sigma**2/2))/sigma**2
    barr_y = np.log(Barrier**2/(S*K))/(sigma*np.sqrt(T)) + barr_lambda*sigma*np.sqrt(T)
    CDI = S*(Barrier/S)**(2*barr_lambda)*norm.cdf(barr_y) - K*np.exp(-r*T)*(Barrier/S)**(2*barr_lambda-2)*norm.cdf(barr_y-sigma*np.sqrt(T))
    
    return     CDI


def BS_Barrier_CDO(S,K,Barrier,T,r,sigma):
    
    barr_lambda = ((r) + (sigma**2/2))/sigma**2
    barr_y = np.log(Barrier**2/(S*K))/(sigma*np.sqrt(T)) + barr_lambda*sigma*np.sqrt(T)
    CDI = S*(Barrier/S)**(2*barr_lambda)*norm.cdf(barr_y) - K*np.exp(-r*T)*(Barrier/S)**(2*barr_lambda-2)*norm.cdf(barr_y-sigma*np.sqrt(T))
    
    CDO = BS_Call_E(S, K, T, r, sigma) - CDI

    return     CDO