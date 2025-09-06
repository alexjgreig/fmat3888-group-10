import numpy as np
from math import log, sqrt, exp, erf

# Black Scholes for validation
def norm_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def bs_call(S0, K, r, sigma, T):
    if T <= 0:
        return max(S0 - K, 0.0)
    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S0 * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)


# Monte Carlo pricing mechanism 
#       (given the simulated paths/terminal prices, price the option)

def mc_european_call_price(S_T, K, r, T):
    payoffs = np.maximum(S_T - K, 0.0)
    disc = np.exp(-r * T)
    vals = disc * payoffs
    price = vals.mean()
    se = vals.std(ddof=1) / np.sqrt(vals.size)
    ci95 = (price - 1.96 * se, price + 1.96 * se)
    return price, se, ci95


# Euler simulation for terminal price

def simulate_terminal_euler(S0, r, sigma, T, steps, Z):
    """
    Simulate terminal S_T prices with Euler discretisation using provided normals Z (n_paths x steps).
    Returns a (n_paths sized) array of S_T prices
    """
    dt = T / steps
    sqrt_dt = np.sqrt(dt)
    S = np.full(Z.shape[0], S0, dtype=float)
    for n in range(steps):
        S += r * S * dt + sigma * S * sqrt_dt * Z[:, n]
    return S

# Milstein simulation for terminal price

def simulate_terminal_milstein(S0, r, sigma, T, steps, Z):
    """
    Simulate terminal S_T prrices with Milstein discretisation using provided normals Z (n_paths x steps).
    Returns a (n_paths sized) array of S_T prices
    """
    dt = T / steps
    sqrt_dt = np.sqrt(dt)
    S = np.full(Z.shape[0], S0, dtype=float)
    for n in range(steps):
        dW = sqrt_dt * Z[:, n]
        S += r * S * dt + sigma * S * dW + 0.5 * (sigma**2) * S * (dW**2 - dt)
    return S


# Execution
#       TO DO: Plug in Yahoo finance data

if __name__ == "__main__":
        S0, K, r, sigma, T = 100.0, 100.0, 0.07, 0.20, 1.0 #Example (replace with yh data)

        # Example parameters (simulate through a range of these to demonstrate convergence)
        steps   = 252       # time steps per year (increase to reduce discretisation bias)
        n_paths = 100_000   # number of paths (increase to reduce MC variance)
        seed    = 12345

        rng = np.random.default_rng(seed)
        Z = rng.standard_normal((n_paths, steps)) # Generate random numbers (to use in both simulations)

        # Euler simulation and MC pricing
        ST_euler = simulate_terminal_euler(S0, r, sigma, T, steps, Z)
        price_euler, se_euler, ci_euler = mc_european_call_price(ST_euler, K, r, T)

        # Milstein simulation and MC pricing
        ST_milst = simulate_terminal_milstein(S0, r, sigma, T, steps, Z)
        price_milst, se_milst, ci_milst = mc_european_call_price(ST_milst, K, r, T)
        
        # BS price for validation
        price_bs = bs_call(S0, K, r, sigma, T)

        # Display results
        print(f"Euler MC: {price_euler:.4f}  (SE={se_euler:.4f}, 95% CI={ci_euler})")
        print(f"Milstein MC      : {price_milst:.4f}  (SE={se_milst:.4f}, 95% CI={ci_milst})")
        print(f"Black–Scholes    : {price_bs:.4f}")







