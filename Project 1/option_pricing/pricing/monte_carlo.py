"""
Monte Carlo simulation engine with Euler and Milstein schemes.
Includes variance reduction techniques and parallel processing.
"""

import numpy as np
from typing import Optional, Tuple, Callable, Dict, Any
from numba import jit, prange
import sobol_seq
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.stats import norm


class MonteCarloEngine:
    """Monte Carlo simulation engine for option pricing."""
    
    def __init__(self, n_paths: int = 100000, n_steps: int = 252,
                 scheme: str = 'milstein', seed: Optional[int] = None,
                 use_antithetic: bool = True, use_control_variate: bool = False,
                 use_quasi_random: bool = False, n_jobs: int = -1):
        """
        Initialize Monte Carlo engine.
        
        Args:
            n_paths: Number of simulation paths
            n_steps: Number of time steps
            scheme: Discretization scheme ('euler' or 'milstein')
            seed: Random seed for reproducibility
            use_antithetic: Use antithetic variates
            use_control_variate: Use control variates
            use_quasi_random: Use quasi-random (Sobol) sequences
            n_jobs: Number of parallel jobs (-1 for all CPUs)
        """
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.scheme = scheme.lower()
        self.seed = seed
        self.use_antithetic = use_antithetic
        self.use_control_variate = use_control_variate
        self.use_quasi_random = use_quasi_random
        self.n_jobs = n_jobs
        
        if seed is not None:
            np.random.seed(seed)
    
    def generate_random_numbers(self, shape: Tuple[int, ...]) -> np.ndarray:
        """
        Generate random numbers using standard or quasi-random sequences.
        
        Args:
            shape: Shape of the random array
            
        Returns:
            Array of random numbers
        """
        if self.use_quasi_random:
            # Use Sobol sequence
            n_samples = np.prod(shape)
            n_dims = shape[-1] if len(shape) > 1 else 1
            
            # Generate Sobol points
            sobol_points = sobol_seq.i4_sobol_generate(n_dims, n_samples)
            
            # Transform to normal distribution
            normal_points = norm.ppf(sobol_points)
            
            return normal_points.reshape(shape)
        else:
            # Use standard pseudo-random numbers
            return np.random.standard_normal(shape)
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _euler_scheme_paths(S0: float, r: float, sigma: float, T: float,
                           n_paths: int, n_steps: int, dW: np.ndarray) -> np.ndarray:
        """
        Generate paths using Euler scheme (JIT compiled).
        
        Args:
            S0: Initial stock price
            r: Risk-free rate
            sigma: Constant volatility
            T: Time to maturity
            n_paths: Number of paths
            n_steps: Number of time steps
            dW: Brownian increments
            
        Returns:
            Array of price paths
        """
        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = S0
        
        for i in prange(n_paths):
            for j in range(n_steps):
                paths[i, j + 1] = paths[i, j] * (1 + r * dt + sigma * dW[i, j])
        
        return paths
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _milstein_scheme_paths(S0: float, r: float, sigma: float, T: float,
                              n_paths: int, n_steps: int, dW: np.ndarray) -> np.ndarray:
        """
        Generate paths using Milstein scheme (JIT compiled).
        
        Args:
            S0: Initial stock price
            r: Risk-free rate
            sigma: Constant volatility
            T: Time to maturity
            n_paths: Number of paths
            n_steps: Number of time steps
            dW: Brownian increments
            
        Returns:
            Array of price paths
        """
        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = S0
        
        for i in prange(n_paths):
            for j in range(n_steps):
                dW_ij = dW[i, j]
                paths[i, j + 1] = paths[i, j] * (
                    1 + r * dt + sigma * dW_ij + 
                    0.5 * sigma**2 * (dW_ij**2 - dt)
                )
        
        return paths
    
    def simulate_gbm_paths(self, S0: float, r: float, sigma: float, T: float,
                          q: float = 0, use_exact: bool = False) -> np.ndarray:
        """
        Simulate Geometric Brownian Motion paths.
        
        Args:
            S0: Initial stock price
            r: Risk-free rate
            sigma: Volatility
            T: Time to maturity
            q: Dividend yield
            use_exact: If True, use exact GBM solution instead of discretization
            
        Returns:
            Array of simulated price paths
        """
        dt = T / self.n_steps
        
        # Generate random increments
        if self.use_antithetic:
            # Generate half the paths
            half_paths = self.n_paths // 2
            dW_half = self.generate_random_numbers((half_paths, self.n_steps))
            dW_half *= np.sqrt(dt)
            
            # Create antithetic pairs
            dW = np.vstack([dW_half, -dW_half])
        else:
            dW = self.generate_random_numbers((self.n_paths, self.n_steps))
            dW *= np.sqrt(dt)
        
        # Adjust drift for dividends
        r_adj = r - q
        
        if use_exact:
            # Use exact GBM solution: S_t = S_0 * exp((r - q - 0.5*sigma^2)*t + sigma*W_t)
            paths = np.zeros((self.n_paths, self.n_steps + 1))
            paths[:, 0] = S0
            
            # Cumulative sum of Brownian increments
            W = np.cumsum(dW, axis=1)
            
            for i in range(self.n_steps):
                t = (i + 1) * dt
                paths[:, i + 1] = S0 * np.exp(
                    (r_adj - 0.5 * sigma**2) * t + sigma * W[:, i]
                )
        else:
            # Generate paths based on discretization scheme
            if self.scheme == 'euler':
                paths = self._euler_scheme_paths(S0, r_adj, sigma, T, 
                                                self.n_paths, self.n_steps, dW)
            elif self.scheme == 'milstein':
                paths = self._milstein_scheme_paths(S0, r_adj, sigma, T,
                                                   self.n_paths, self.n_steps, dW)
            else:
                raise ValueError(f"Unknown scheme: {self.scheme}")
        
        return paths
    
    def simulate_local_vol_paths(self, S0: float, r: float, T: float,
                                local_vol_func: Callable[[float, float], float],
                                q: float = 0, use_log_scheme: bool = False) -> np.ndarray:
        """
        Simulate paths with local volatility model.
        
        Args:
            S0: Initial stock price
            r: Risk-free rate
            T: Time to maturity
            local_vol_func: Function that returns local vol given (S, t)
            q: Dividend yield
            use_log_scheme: If True, simulate in log-space for better stability
            
        Returns:
            Array of simulated price paths
        """
        dt = T / self.n_steps
        paths = np.zeros((self.n_paths, self.n_steps + 1))
        paths[:, 0] = S0
        
        # Generate random increments
        if self.use_antithetic:
            half_paths = self.n_paths // 2
            dW_half = self.generate_random_numbers((half_paths, self.n_steps))
            dW_half *= np.sqrt(dt)
            dW = np.vstack([dW_half, -dW_half])
        else:
            dW = self.generate_random_numbers((self.n_paths, self.n_steps))
            dW *= np.sqrt(dt)
        
        # Adjust drift for dividends
        r_adj = r - q
        
        if use_log_scheme:
            # Simulate in log-space for better numerical stability
            log_paths = np.zeros((self.n_paths, self.n_steps + 1))
            log_paths[:, 0] = np.log(S0)
            
            for i in range(self.n_steps):
                # Use the midpoint time for better accuracy
                t_current = i * dt
                t_next = (i + 1) * dt
                t_mid = 0.5 * (t_current + t_next)
                
                for j in range(self.n_paths):
                    S = np.exp(log_paths[j, i])
                    # Use midpoint time for local volatility evaluation
                    sigma_local = local_vol_func(S, t_mid)
                    
                    # Log-space SDE: d(log S) = (r - q - 0.5*sigma^2)dt + sigma*dW
                    log_paths[j, i + 1] = log_paths[j, i] + \
                        (r_adj - 0.5 * sigma_local**2) * dt + \
                        sigma_local * dW[j, i]
            
            # Convert back to price space
            paths = np.exp(log_paths)
        else:
            # Standard simulation
            for i in range(self.n_steps):
                # Use the midpoint time for better accuracy
                t_current = i * dt
                t_next = (i + 1) * dt
                t_mid = 0.5 * (t_current + t_next)
                
                for j in range(self.n_paths):
                    S = paths[j, i]
                    # Use midpoint time for local volatility evaluation
                    sigma_local = local_vol_func(S, t_mid)
                    
                    if self.scheme == 'euler':
                        # Correct Euler scheme: dS = S*r*dt + S*sigma*dW
                        paths[j, i + 1] = S + S * r_adj * dt + S * sigma_local * dW[j, i]
                    elif self.scheme == 'milstein':
                        # For local vol, we need the derivative of sigma w.r.t. S
                        # Approximate using central finite difference
                        dS = max(S * 0.001, 0.01)  # Ensure minimum perturbation
                        
                        # Central difference for better accuracy
                        if S > dS:
                            sigma_up = local_vol_func(S + dS, t_mid)
                            sigma_down = local_vol_func(S - dS, t_mid)
                            dsigma_dS = (sigma_up - sigma_down) / (2 * dS)
                        else:
                            # Forward difference near zero
                            sigma_up = local_vol_func(S + dS, t_mid)
                            dsigma_dS = (sigma_up - sigma_local) / dS
                        
                        # Correct Milstein scheme for local volatility
                        # dS = S*r*dt + S*sigma*dW + 0.5*S^2*sigma*dsigma/dS*(dW^2 - dt)
                        paths[j, i + 1] = S + S * r_adj * dt + S * sigma_local * dW[j, i] + \
                                         0.5 * S**2 * sigma_local * dsigma_dS * (dW[j, i]**2 - dt)
        
        return paths
    
    def price_european_option(self, S0: float, K: float, T: float, r: float,
                            sigma: float, option_type: str = 'call',
                            q: float = 0) -> Dict[str, float]:
        """
        Price European option using Monte Carlo.
        
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            q: Dividend yield
            
        Returns:
            Dictionary with price and standard error
        """
        # Simulate paths
        paths = self.simulate_gbm_paths(S0, r, sigma, T, q)
        
        # Calculate payoffs
        ST = paths[:, -1]
        if option_type.lower() == 'call':
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)
        
        # Apply control variate if requested
        if self.use_control_variate:
            # Use geometric average as control variate
            geo_avg = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))
            
            # Analytical price of geometric average option
            sigma_geo = sigma / np.sqrt(3)
            r_geo = 0.5 * (r - q - sigma**2 / 6)
            
            from ..models.black_scholes import BlackScholes
            if option_type.lower() == 'call':
                control_price = BlackScholes.call_price(S0, K, T, r_geo, sigma_geo, q)
            else:
                control_price = BlackScholes.put_price(S0, K, T, r_geo, sigma_geo, q)
            
            # Calculate control variate payoffs
            if option_type.lower() == 'call':
                control_payoffs = np.maximum(geo_avg - K, 0)
            else:
                control_payoffs = np.maximum(K - geo_avg, 0)
            
            # Apply control variate correction
            cov = np.cov(payoffs, control_payoffs)[0, 1]
            var_control = np.var(control_payoffs)
            
            if var_control > 0:
                beta = cov / var_control
                payoffs = payoffs - beta * (control_payoffs - control_price * np.exp(r * T))
        
        # Discount to present value
        discounted_payoffs = np.exp(-r * T) * payoffs
        
        # Calculate price and standard error
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(self.n_paths)
        
        return {
            'price': price,
            'std_error': std_error,
            'confidence_interval': (price - 1.96 * std_error, price + 1.96 * std_error)
        }
    
    def price_barrier_option(self, S0: float, K: float, H: float, T: float,
                           r: float, sigma: float, barrier_type: str = 'down-out',
                           option_type: str = 'call', q: float = 0) -> Dict[str, float]:
        """
        Price barrier option using Monte Carlo.
        
        Args:
            S0: Initial stock price
            K: Strike price
            H: Barrier level
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            barrier_type: 'down-out', 'down-in', 'up-out', 'up-in'
            option_type: 'call' or 'put'
            q: Dividend yield
            
        Returns:
            Dictionary with price, standard error, and knock probability
        """
        # Simulate paths
        paths = self.simulate_gbm_paths(S0, r, sigma, T, q)
        
        # Check barrier conditions
        if 'down' in barrier_type:
            if 'out' in barrier_type:
                # Down-and-out: knocked out if any price <= H
                knocked_out = np.any(paths <= H, axis=1)
                active = ~knocked_out
            else:
                # Down-and-in: knocked in if any price <= H
                knocked_in = np.any(paths <= H, axis=1)
                active = knocked_in
        else:  # 'up' in barrier_type
            if 'out' in barrier_type:
                # Up-and-out: knocked out if any price >= H
                knocked_out = np.any(paths >= H, axis=1)
                active = ~knocked_out
            else:
                # Up-and-in: knocked in if any price >= H
                knocked_in = np.any(paths >= H, axis=1)
                active = knocked_in
        
        # Calculate payoffs for active paths
        ST = paths[:, -1]
        if option_type.lower() == 'call':
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)
        
        # Apply barrier condition
        payoffs = payoffs * active
        
        # Discount to present value
        discounted_payoffs = np.exp(-r * T) * payoffs
        
        # Calculate price and statistics
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(self.n_paths)
        knock_probability = np.mean(active)
        
        return {
            'price': price,
            'std_error': std_error,
            'confidence_interval': (price - 1.96 * std_error, price + 1.96 * std_error),
            'knock_probability': knock_probability
        }
    
    def estimate_convergence(self, pricing_func: Callable, 
                           path_counts: list = None,
                           **pricing_params) -> Dict[str, Any]:
        """
        Estimate convergence of Monte Carlo pricing.
        
        Args:
            pricing_func: Pricing function to test
            path_counts: List of path counts to test
            **pricing_params: Parameters for pricing function
            
        Returns:
            Dictionary with convergence results
        """
        if path_counts is None:
            path_counts = [1000, 5000, 10000, 50000, 100000, 500000]
        
        results = {'path_counts': path_counts, 'prices': [], 'std_errors': []}
        
        original_n_paths = self.n_paths
        
        for n in tqdm(path_counts, desc="Testing convergence"):
            self.n_paths = n
            result = pricing_func(**pricing_params)
            results['prices'].append(result['price'])
            results['std_errors'].append(result['std_error'])
        
        # Restore original setting
        self.n_paths = original_n_paths
        
        # Calculate convergence rate
        results['prices'] = np.array(results['prices'])
        results['std_errors'] = np.array(results['std_errors'])
        
        # Theoretical convergence rate for Monte Carlo is O(1/sqrt(n))
        theoretical_errors = results['std_errors'][0] * np.sqrt(path_counts[0] / np.array(path_counts))
        results['theoretical_errors'] = theoretical_errors
        
        return results