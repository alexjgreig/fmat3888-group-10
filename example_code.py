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


def Milstein(S, K, T, R, sigma, M, N):
    
    # grid points
    T_INIT = 0
    T_END = T

    DT = float(T_END - T_INIT) / M
    TS = np.arange(T_INIT, T_END + DT, DT)
   
    Y_INIT = 1

    # Vectors to fill
    ys = np.zeros(M + 1)
    V  = np.zeros(N)
    ys[0] = S
    mu = 0
    
    for j in range(N):
        for i in range(1, TS.size):
                t = (i - 1) * DT
                y = ys[i - 1]
                dw =  np.random.normal(loc=0.0, scale=np.sqrt(DT)) 
        
                # Sum up terms as in the Milstein method 
                ys[i] = y + R * y * DT + sigma * y * dw + (sigma**2 / 2) * y * (dw**2 - DT) 

        if np.min(ys) <= Barrier :
            a = 0
        else:
            a = 1 
        mu = mu + a    

        
        V[j] =  np.maximum(ys[TS.size-1] - K,0) * a
    
    return  V, mu


#def plot_simulations(num_sims: int):
  
VV = np.zeros(N)
for i in range(1,N):
    price, mu = Milstein(S, K, T, R, sigma, M, N)
    VV[i] =  np.sum(price[0:i])  /  i


OptionPrice = np.exp(-R*T) * VV[N-1]

mu = np.exp(-R*T) * (1- mu / N    )

print("Option Price - ",round(OptionPrice,6)," with Barrier - ", Barrier)
print("Probability of hitting barrier - ",round(mu*100,2),"%")
BSE_CDO_Price = BS_Barrier_CDO(S,K,Barrier,T,R, sigma)
print("Barrier BS - ", BSE_CDO_Price)

plt.plot(VV, color = 'red')

plt.plot(range(N), [BSE_CDO_Price for i in range(N)] ,linestyle='--', color='g')  # color='green' , marker='o'

#plt.xlabel("time (s)")
plt.ylabel("Option Price")
plt.grid()
plt.show()


#Euler Scheme

Asset_At_Exp = np.zeros(N)
mu = 0

# grid points
T_INIT = 0
T_END = T

DT = float(T_END - T_INIT) / M

for i in range(N):
    
        Asset = np.zeros(M+1)
        Asset[0] = S

        xi = np.random.normal(0, 1, M)
        for k in range(M):
            Asset[k+1] = Asset[k] + Asset[k] * R * DT + Asset[k] * sigma * np.sqrt(DT) * xi[k]

        if np.min(Asset) <= Barrier:
            a = 0
        else:
            a = 1
            
        mu = mu + a    

        Asset[k+1] =  np.maximum(Asset[k+1]-K,0)*a
        
        Asset_At_Exp[i] = Asset[k+1]


OptionPrice = np.exp(-R*T) * np.sum(Asset_At_Exp) / N 

mu = np.exp(-R*T) * (1- mu / N    )
print("Option Price - ",round(OptionPrice,6)," with Barrier - ", Barrier)
print("Probability of hitting barrier - ",round(mu*100,2),"%")
print("Barrier BS - ", BS_Barrier_CDO(S,K,Barrier,T,R,sigma))

