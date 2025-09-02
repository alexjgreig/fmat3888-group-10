"""
Black-Scholes model implementation with Greeks and implied volatility.
"""

import numpy as np
from scipy.stats import norm
from typing import Union, Optional
from scipy.optimize import brentq, minimize_scalar


class BlackScholes:
    """Black-Scholes model for European and Barrier options."""
    
    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        """Calculate d1 parameter."""
        return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        """Calculate d2 parameter."""
        return BlackScholes.d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)
    
    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        """
        Calculate European call option price.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield (default 0)
            
        Returns:
            Call option price
        """
        if T <= 0:
            return max(S - K, 0)
        
        d1 = BlackScholes.d1(S, K, T, r, sigma, q)
        d2 = BlackScholes.d2(S, K, T, r, sigma, q)
        
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        """
        Calculate European put option price.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield (default 0)
            
        Returns:
            Put option price
        """
        if T <= 0:
            return max(K - S, 0)
        
        d1 = BlackScholes.d1(S, K, T, r, sigma, q)
        d2 = BlackScholes.d2(S, K, T, r, sigma, q)
        
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    
    @staticmethod
    def delta(S: float, K: float, T: float, r: float, sigma: float, 
              q: float = 0, option_type: str = 'call') -> float:
        """Calculate option delta."""
        if T <= 0:
            if option_type.lower() == 'call':
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
        
        d1 = BlackScholes.d1(S, K, T, r, sigma, q)
        
        if option_type.lower() == 'call':
            return np.exp(-q * T) * norm.cdf(d1)
        else:
            return -np.exp(-q * T) * norm.cdf(-d1)
    
    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        """Calculate option gamma."""
        if T <= 0:
            return 0.0
        
        d1 = BlackScholes.d1(S, K, T, r, sigma, q)
        return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        """Calculate option vega."""
        if T <= 0:
            return 0.0
        
        d1 = BlackScholes.d1(S, K, T, r, sigma, q)
        return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    
    @staticmethod
    def theta(S: float, K: float, T: float, r: float, sigma: float, 
              q: float = 0, option_type: str = 'call') -> float:
        """Calculate option theta (per day)."""
        if T <= 0:
            return 0.0
        
        d1 = BlackScholes.d1(S, K, T, r, sigma, q)
        d2 = BlackScholes.d2(S, K, T, r, sigma, q)
        
        term1 = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        
        if option_type.lower() == 'call':
            term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
            term3 = q * S * np.exp(-q * T) * norm.cdf(d1)
            theta_annual = term1 + term2 + term3
        else:
            term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
            term3 = -q * S * np.exp(-q * T) * norm.cdf(-d1)
            theta_annual = term1 + term2 + term3
        
        return theta_annual / 365.25  # Convert to per day
    
    @staticmethod
    def rho(S: float, K: float, T: float, r: float, sigma: float, 
            q: float = 0, option_type: str = 'call') -> float:
        """Calculate option rho."""
        if T <= 0:
            return 0.0
        
        d2 = BlackScholes.d2(S, K, T, r, sigma, q)
        
        if option_type.lower() == 'call':
            return K * T * np.exp(-r * T) * norm.cdf(d2)
        else:
            return -K * T * np.exp(-r * T) * norm.cdf(-d2)
    
    @staticmethod
    def vomma(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        """Calculate vomma (volga) - sensitivity of vega to volatility."""
        if T <= 0:
            return 0.0
        
        d1 = BlackScholes.d1(S, K, T, r, sigma, q)
        d2 = BlackScholes.d2(S, K, T, r, sigma, q)
        vega = BlackScholes.vega(S, K, T, r, sigma, q)
        
        return vega * d1 * d2 / sigma
    
    @staticmethod
    def vanna(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        """Calculate vanna - sensitivity of delta to volatility."""
        if T <= 0:
            return 0.0
        
        d1 = BlackScholes.d1(S, K, T, r, sigma, q)
        d2 = BlackScholes.d2(S, K, T, r, sigma, q)
        
        return -np.exp(-q * T) * norm.pdf(d1) * d2 / sigma
    
    @staticmethod
    def implied_volatility(price: float, S: float, K: float, T: float, r: float,
                          q: float = 0, option_type: str = 'call',
                          max_iterations: int = 100, tolerance: float = 1e-6) -> Optional[float]:
        """
        Calculate implied volatility using Newton-Raphson method with Brent fallback.
        
        Args:
            price: Option market price
            S: Current stock price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            q: Dividend yield
            option_type: 'call' or 'put'
            max_iterations: Maximum iterations for Newton-Raphson
            tolerance: Convergence tolerance
            
        Returns:
            Implied volatility or None if not found
        """
        if T <= 0:
            return None
        
        # Check for arbitrage violations
        if option_type.lower() == 'call':
            if price < max(S * np.exp(-q * T) - K * np.exp(-r * T), 0):
                return None
            if price > S * np.exp(-q * T):
                return None
        else:
            if price < max(K * np.exp(-r * T) - S * np.exp(-q * T), 0):
                return None
            if price > K * np.exp(-r * T):
                return None
        
        # Initial guess using Brenner-Subrahmanyam approximation
        sigma = np.sqrt(2 * np.pi / T) * price / S
        sigma = max(0.01, min(sigma, 5.0))  # Bound initial guess
        
        # Newton-Raphson iteration
        for i in range(max_iterations):
            if option_type.lower() == 'call':
                model_price = BlackScholes.call_price(S, K, T, r, sigma, q)
            else:
                model_price = BlackScholes.put_price(S, K, T, r, sigma, q)
            
            vega = BlackScholes.vega(S, K, T, r, sigma, q)
            
            if abs(vega) < 1e-10:
                break
            
            diff = price - model_price
            
            if abs(diff) < tolerance:
                return sigma
            
            sigma = sigma + diff / vega
            sigma = max(0.001, min(sigma, 10.0))  # Keep sigma in reasonable bounds
        
        # If Newton-Raphson fails, try Brent's method
        try:
            def objective(vol):
                if option_type.lower() == 'call':
                    return BlackScholes.call_price(S, K, T, r, vol, q) - price
                else:
                    return BlackScholes.put_price(S, K, T, r, vol, q) - price
            
            sigma = brentq(objective, 0.001, 10.0, xtol=tolerance, maxiter=max_iterations)
            return sigma
        except:
            return None
    
    @staticmethod
    def barrier_call_down_out(S: float, K: float, H: float, T: float, 
                             r: float, sigma: float, q: float = 0) -> float:
        """
        Calculate down-and-out call option price.
        
        Args:
            S: Current stock price
            K: Strike price
            H: Barrier level (H < S)
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield
            
        Returns:
            Down-and-out call option price
        """
        if H >= S:
            return 0.0
        
        if T <= 0:
            return max(S - K, 0) if S > H else 0.0
        
        # Lambda parameter
        lam = (r - q + 0.5 * sigma**2) / (sigma**2)
        
        # y parameter
        y = np.log(H**2 / (S * K)) / (sigma * np.sqrt(T)) + lam * sigma * np.sqrt(T)
        
        # Calculate CDI (down-and-in call)
        cdi = (S * np.exp(-q * T) * (H / S)**(2 * lam) * norm.cdf(y) - 
               K * np.exp(-r * T) * (H / S)**(2 * lam - 2) * norm.cdf(y - sigma * np.sqrt(T)))
        
        # CDO = Vanilla Call - CDI
        vanilla_call = BlackScholes.call_price(S, K, T, r, sigma, q)
        return vanilla_call - cdi
    
    @staticmethod
    def barrier_call_down_in(S: float, K: float, H: float, T: float, 
                            r: float, sigma: float, q: float = 0) -> float:
        """
        Calculate down-and-in call option price.
        
        Args:
            S: Current stock price
            K: Strike price
            H: Barrier level (H < S)
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield
            
        Returns:
            Down-and-in call option price
        """
        if H >= S:
            return BlackScholes.call_price(S, K, T, r, sigma, q)
        
        if T <= 0:
            return 0.0
        
        # Lambda parameter
        lam = (r - q + 0.5 * sigma**2) / (sigma**2)
        
        # y parameter
        y = np.log(H**2 / (S * K)) / (sigma * np.sqrt(T)) + lam * sigma * np.sqrt(T)
        
        # Calculate CDI (down-and-in call)
        cdi = (S * np.exp(-q * T) * (H / S)**(2 * lam) * norm.cdf(y) - 
               K * np.exp(-r * T) * (H / S)**(2 * lam - 2) * norm.cdf(y - sigma * np.sqrt(T)))
        
        return cdi
    
    @staticmethod
    def barrier_put_up_out(S: float, K: float, H: float, T: float, 
                          r: float, sigma: float, q: float = 0) -> float:
        """
        Calculate up-and-out put option price.
        
        Args:
            S: Current stock price
            K: Strike price
            H: Barrier level (H > S)
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield
            
        Returns:
            Up-and-out put option price
        """
        if H <= S:
            return 0.0
        
        if T <= 0:
            return max(K - S, 0) if S < H else 0.0
        
        # Lambda parameter
        lam = (r - q + 0.5 * sigma**2) / (sigma**2)
        
        # y parameter
        y = np.log(H**2 / (S * K)) / (sigma * np.sqrt(T)) + lam * sigma * np.sqrt(T)
        
        # Calculate PUI (up-and-in put)
        pui = (-S * np.exp(-q * T) * (H / S)**(2 * lam) * norm.cdf(-y) + 
               K * np.exp(-r * T) * (H / S)**(2 * lam - 2) * norm.cdf(-y + sigma * np.sqrt(T)))
        
        # PUO = Vanilla Put - PUI
        vanilla_put = BlackScholes.put_price(S, K, T, r, sigma, q)
        return vanilla_put - pui
    
    @staticmethod
    def digital_call(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        """Calculate digital (binary) call option price."""
        if T <= 0:
            return 1.0 if S > K else 0.0
        
        d2 = BlackScholes.d2(S, K, T, r, sigma, q)
        return np.exp(-r * T) * norm.cdf(d2)
    
    @staticmethod
    def digital_put(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        """Calculate digital (binary) put option price."""
        if T <= 0:
            return 1.0 if S < K else 0.0
        
        d2 = BlackScholes.d2(S, K, T, r, sigma, q)
        return np.exp(-r * T) * norm.cdf(-d2)