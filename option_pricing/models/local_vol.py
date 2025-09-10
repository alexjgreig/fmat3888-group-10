"""
Dupire Local Volatility Model implementation.
Reference: B. Dupire, "Pricing with a smile", Risk, 7 (1994), pp. 18–20.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline, interp1d
from typing import Optional, Callable, Dict, Tuple
from ..models.volatility_surface import ParametricVolatilitySurface
from ..models.black_scholes import BlackScholes
from ..pricing.monte_carlo import MonteCarloEngine


class DupireLocalVolatility:
    """
    Dupire Local Volatility Model.
    
    The local volatility σ_loc(S,t) is derived from the implied volatility surface
    using Dupire's formula.
    """
    
    def __init__(self, vol_surface: Optional[ParametricVolatilitySurface] = None,
                 spot: float = None, risk_free_rate: float = 0.05,
                 dividend_yield: float = 0.0):
        """
        Initialize Dupire local volatility model.
        
        Args:
            vol_surface: Calibrated volatility surface
            spot: Current spot price
            risk_free_rate: Risk-free interest rate
            dividend_yield: Dividend yield
        """
        self.vol_surface = vol_surface
        self.spot = spot
        self.r = risk_free_rate
        self.q = dividend_yield
        self.local_vol_grid = None
        self.local_vol_interpolator = None
        
    def dupire_formula(self, K: float, T: float, dK: float = 0.01, dT: float = 0.001) -> float:
        """
        Calculate local volatility using Dupire's formula.
        
        σ_loc²(K,T) = (∂C/∂T + (r-q)K∂C/∂K + qC) / (0.5K²∂²C/∂K²)
        
        Where C is the call option price.
        
        Args:
            K: Strike price
            T: Time to maturity
            dK: Strike increment for numerical derivatives
            dT: Time increment for numerical derivatives
            
        Returns:
            Local volatility at (K, T)
        """
        if T <= 0:
            return self.vol_surface.get_vol(K, 0.001) if self.vol_surface else 0.2
        
        # Get implied volatilities
        vol = self.vol_surface.get_vol(K, T)
        
        # Calculate call prices using Black-Scholes
        C = BlackScholes.call_price(self.spot, K, T, self.r, vol, self.q)
        
        # Numerical derivatives
        # ∂C/∂T
        if T > dT:
            vol_T_plus = self.vol_surface.get_vol(K, T + dT)
            vol_T_minus = self.vol_surface.get_vol(K, T - dT)
            C_T_plus = BlackScholes.call_price(self.spot, K, T + dT, self.r, vol_T_plus, self.q)
            C_T_minus = BlackScholes.call_price(self.spot, K, T - dT, self.r, vol_T_minus, self.q)
            dC_dT = (C_T_plus - C_T_minus) / (2 * dT)
        else:
            vol_T_plus = self.vol_surface.get_vol(K, T + dT)
            C_T_plus = BlackScholes.call_price(self.spot, K, T + dT, self.r, vol_T_plus, self.q)
            dC_dT = (C_T_plus - C) / dT
        
        # ∂C/∂K
        vol_K_plus = self.vol_surface.get_vol(K + dK, T)
        vol_K_minus = self.vol_surface.get_vol(K - dK, T)
        C_K_plus = BlackScholes.call_price(self.spot, K + dK, T, self.r, vol_K_plus, self.q)
        C_K_minus = BlackScholes.call_price(self.spot, K - dK, T, self.r, vol_K_minus, self.q)
        dC_dK = (C_K_plus - C_K_minus) / (2 * dK)
        
        # ∂²C/∂K²
        d2C_dK2 = (C_K_plus - 2 * C + C_K_minus) / (dK**2)
        
        # Apply Dupire formula
        numerator = dC_dT + (self.r - self.q) * K * dC_dK + self.q * C
        denominator = 0.5 * K**2 * d2C_dK2
        
        # Handle edge cases
        if denominator <= 0 or numerator < 0:
            # Fall back to implied volatility
            return vol
        
        local_variance = numerator / denominator
        
        if local_variance < 0:
            return vol
        
        return np.sqrt(local_variance)
    
    def build_local_vol_surface(self, strike_range: Tuple[float, float] = (0.5, 2.0),
                              maturity_range: Tuple[float, float] = (0.01, 2.0),
                              n_strikes: int = 50, n_maturities: int = 30):
        """
        Build a grid of local volatilities.
        
        Args:
            strike_range: Range of strikes as fraction of spot
            maturity_range: Range of maturities in years
            n_strikes: Number of strike points
            n_maturities: Number of maturity points
        """
        if self.vol_surface is None:
            raise ValueError("Volatility surface not set. Calibrate or set a surface first.")
        
        # Create grids
        strikes = np.linspace(
            self.spot * strike_range[0],
            self.spot * strike_range[1],
            n_strikes
        )
        maturities = np.linspace(maturity_range[0], maturity_range[1], n_maturities)
        
        # Calculate local volatility grid
        local_vol_grid = np.zeros((n_strikes, n_maturities))
        
        print("Building local volatility surface...")
        for i, K in enumerate(strikes):
            for j, T in enumerate(maturities):
                local_vol_grid[i, j] = self.dupire_formula(K, T)
        
        self.local_vol_grid = {
            'strikes': strikes,
            'maturities': maturities,
            'values': local_vol_grid
        }
        
        # Create interpolator for fast evaluation
        self.local_vol_interpolator = RectBivariateSpline(
            strikes, maturities, local_vol_grid, kx=3, ky=1
        )
        
        print(f"Local volatility surface built: {n_strikes} strikes × {n_maturities} maturities")
    
    def get_local_vol(self, S: float, t: float) -> float:
        """
        Get local volatility at given spot and time.
        
        Args:
            S: Spot price
            t: Time
            
        Returns:
            Local volatility
        """
        if self.local_vol_interpolator is not None:
            # Use interpolator
            vol = float(self.local_vol_interpolator(S, t))
            return max(vol, 0.01)  # Ensure positive volatility
        elif self.vol_surface is not None:
            # Use Dupire formula directly
            return self.dupire_formula(S, t)
        else:
            # Default constant volatility
            return 0.2
    
    def price_european_option(self, K: float, T: float, option_type: str = 'call',
                            n_paths: int = 100000, n_steps: int = 100,
                            scheme: str = 'milstein', use_log_scheme: bool = True) -> Dict:
        """
        Price European option using local volatility Monte Carlo.
        
        Args:
            K: Strike price
            T: Time to maturity
            option_type: 'call' or 'put'
            n_paths: Number of Monte Carlo paths
            n_steps: Number of time steps
            scheme: 'euler' or 'milstein'
            use_log_scheme: If True, simulate in log-space for better stability
            
        Returns:
            Dictionary with pricing results
        """
        # Create Monte Carlo engine
        mc_engine = MonteCarloEngine(
            n_paths=n_paths,
            n_steps=n_steps,
            scheme=scheme,
            use_antithetic=True
        )
        
        # Define local vol function
        def local_vol_func(S, t):
            return self.get_local_vol(S, t)
        
        # Simulate paths
        paths = mc_engine.simulate_local_vol_paths(
            self.spot, self.r, T, local_vol_func, self.q, use_log_scheme=use_log_scheme
        )
        
        # Calculate payoffs
        ST = paths[:, -1]
        if option_type.lower() == 'call':
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)
        
        # Discount to present value
        discounted_payoffs = np.exp(-self.r * T) * payoffs
        
        # Calculate price and statistics
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(n_paths)
        
        # Compare with Black-Scholes price using ATM vol
        atm_vol = self.vol_surface.get_vol(K, T) if self.vol_surface else 0.2
        if option_type.lower() == 'call':
            bs_price = BlackScholes.call_price(self.spot, K, T, self.r, atm_vol, self.q)
        else:
            bs_price = BlackScholes.put_price(self.spot, K, T, self.r, atm_vol, self.q)
        
        return {
            'price': price,
            'std_error': std_error,
            'confidence_interval': (price - 1.96 * std_error, price + 1.96 * std_error),
            'bs_price': bs_price,
            'difference': price - bs_price
        }
    
    def price_barrier_option(self, K: float, H: float, T: float,
                           barrier_type: str = 'down-out', option_type: str = 'call',
                           n_paths: int = 100000, n_steps: int = 252,
                           scheme: str = 'milstein', use_log_scheme: bool = True) -> Dict:
        """
        Price barrier option using local volatility Monte Carlo.
        
        Args:
            K: Strike price
            H: Barrier level
            T: Time to maturity
            barrier_type: 'down-out', 'down-in', 'up-out', 'up-in'
            option_type: 'call' or 'put'
            n_paths: Number of Monte Carlo paths
            n_steps: Number of time steps (use more for barrier options)
            scheme: 'euler' or 'milstein'
            use_log_scheme: If True, simulate in log-space for better stability
            
        Returns:
            Dictionary with pricing results
        """
        # Create Monte Carlo engine
        mc_engine = MonteCarloEngine(
            n_paths=n_paths,
            n_steps=n_steps,
            scheme=scheme,
            use_antithetic=True
        )
        
        # Define local vol function
        def local_vol_func(S, t):
            return self.get_local_vol(S, t)
        
        # Simulate paths
        paths = mc_engine.simulate_local_vol_paths(
            self.spot, self.r, T, local_vol_func, self.q, use_log_scheme=use_log_scheme
        )
        
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
        discounted_payoffs = np.exp(-self.r * T) * payoffs
        
        # Calculate price and statistics
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(n_paths)
        knock_probability = np.mean(active)
        
        # Compare with Black-Scholes barrier price
        atm_vol = self.vol_surface.get_vol(self.spot, T) if self.vol_surface else 0.2
        
        if barrier_type == 'down-out' and option_type.lower() == 'call':
            bs_price = BlackScholes.barrier_call_down_out(
                self.spot, K, H, T, self.r, atm_vol, self.q
            )
        elif barrier_type == 'down-in' and option_type.lower() == 'call':
            bs_price = BlackScholes.barrier_call_down_in(
                self.spot, K, H, T, self.r, atm_vol, self.q
            )
        else:
            # For other combinations, use vanilla BS as approximation
            if option_type.lower() == 'call':
                bs_price = BlackScholes.call_price(self.spot, K, T, self.r, atm_vol, self.q)
            else:
                bs_price = BlackScholes.put_price(self.spot, K, T, self.r, atm_vol, self.q)
            bs_price *= knock_probability  # Rough adjustment
        
        return {
            'price': price,
            'std_error': std_error,
            'confidence_interval': (price - 1.96 * std_error, price + 1.96 * std_error),
            'knock_probability': knock_probability,
            'bs_price': bs_price,
            'difference': price - bs_price
        }
    
    def analyze_convergence(self, K: float, T: float, option_type: str = 'call',
                          path_counts: list = None, step_counts: list = None,
                          target_error: float = 1e-6, fast_mode: bool = True) -> Dict:
        """
        Analyze Monte Carlo convergence for local volatility pricing.
        
        Args:
            K: Strike price
            T: Time to maturity
            option_type: 'call' or 'put'
            path_counts: List of path counts to test
            step_counts: List of step counts to test
            target_error: Target error tolerance
            fast_mode: Use faster but less accurate convergence testing
            
        Returns:
            Dictionary with convergence analysis
        """
        if path_counts is None:
            if fast_mode:
                # Reduced path counts for faster execution
                path_counts = [1000, 2500, 5000, 10000, 25000]
            else:
                path_counts = [1000, 5000, 10000, 50000, 100000, 500000]
        
        if step_counts is None:
            if fast_mode:
                # Fewer step counts to test
                step_counts = [25, 50, 100]
            else:
                step_counts = [50, 100, 252, 500]
        
        results = {
            'path_convergence': {},
            'step_convergence': {},
            'achieved_error': None,
            'required_paths': None
        }
        
        # Use simplified Monte Carlo for convergence testing
        mc_engine = MonteCarloEngine(
            n_paths=1000,
            n_steps=50 if fast_mode else 100,
            scheme='euler',  # Euler is faster than Milstein
            use_antithetic=True,  # Keep variance reduction
            use_quasi_random=False  # Standard random is faster
        )
        
        # Test convergence w.r.t. number of paths
        print("Testing path convergence...")
        prices = []
        errors = []
        
        # Use constant volatility approximation for speed in fast mode
        if fast_mode and self.vol_surface:
            # Use ATM vol as approximation
            const_vol = self.vol_surface.get_vol(K, T)
            
            for n_paths in path_counts:
                mc_engine.n_paths = n_paths
                result = mc_engine.price_european_option(
                    self.spot, K, T, self.r, const_vol, option_type, self.q
                )
                prices.append(result['price'])
                errors.append(result['std_error'])
                
                # Early stopping if target achieved
                if result['std_error'] <= target_error:
                    results['achieved_error'] = result['std_error']
                    results['required_paths'] = n_paths
                    print(f"Target error {target_error} achieved with {n_paths} paths")
                    # Truncate remaining path counts
                    path_counts = path_counts[:len(prices)]
                    break
        else:
            # Full local vol pricing (slower)
            for n_paths in path_counts:
                result = self.price_european_option(
                    K, T, option_type, n_paths=n_paths, 
                    n_steps=50 if fast_mode else 252
                )
                prices.append(result['price'])
                errors.append(result['std_error'])
                
                # Early stopping
                if result['std_error'] <= target_error:
                    results['achieved_error'] = result['std_error']
                    results['required_paths'] = n_paths
                    print(f"Target error {target_error} achieved with {n_paths} paths")
                    path_counts = path_counts[:len(prices)]
                    break
        
        results['path_convergence'] = {
            'path_counts': path_counts,
            'prices': prices,
            'std_errors': errors
        }
        
        # Skip step convergence in fast mode or if we've already achieved target
        if not fast_mode and not results['achieved_error']:
            print("Testing step convergence...")
            prices = []
            
            # Use fewer paths for step convergence testing
            test_paths = min(10000, path_counts[-1] // 2)
            
            for n_steps in step_counts:
                result = self.price_european_option(
                    K, T, option_type, n_paths=test_paths, n_steps=n_steps
                )
                prices.append(result['price'])
            
            results['step_convergence'] = {
                'step_counts': step_counts,
                'prices': prices
            }
        else:
            # Provide minimal step convergence data
            results['step_convergence'] = {
                'step_counts': [50],
                'prices': [prices[-1] if prices else 0]
            }
        
        # If target not achieved, estimate required paths
        if not results['achieved_error'] and errors:
            # Monte Carlo error scales as 1/sqrt(n)
            last_error = errors[-1]
            last_n = path_counts[-1]
            estimated_n = int(last_n * (last_error / target_error)**2)
            print(f"Estimated paths needed for target error: {estimated_n:,}")
            results['estimated_paths'] = estimated_n
        
        return results
    

    def plot_barrier_paths(self, K: float, H: float, T: float,
                        barrier_type: str = 'down-out', option_type: str = 'call',
                        n_paths: int = 200, n_steps: int = 252,
                        scheme: str = 'milstein', use_log_scheme: bool = True):
        """
        Plot simulated paths for a barrier option with barrier and average path.
        
        Args:
            K: Strike price
            H: Barrier level
            T: Time to maturity
            barrier_type: 'down-out', 'down-in', 'up-out', 'up-in'
            option_type: 'call' or 'put'
            n_paths: Number of simulated paths (use small number for visualization)
            n_steps: Number of time steps
            scheme: 'euler' or 'milstein'
            use_log_scheme: If True, simulate in log-space
        """
        import matplotlib.pyplot as plt
        # Monte Carlo simulation
        mc_engine = MonteCarloEngine(
            n_paths=n_paths,
            n_steps=n_steps,
            scheme=scheme,
            use_antithetic=True
        )

        def local_vol_func(S, t):
            return self.get_local_vol(S, t)

        paths = mc_engine.simulate_local_vol_paths(
            self.spot, self.r, T, local_vol_func, self.q, use_log_scheme=use_log_scheme
        )

        # Time grid
        time_grid = np.linspace(0, T, n_steps + 1)

        cmap = plt.get_cmap('turbo')                 # or 'rainbow' / 'nipy_spectral'
        colors = cmap(np.linspace(0, 1, n_paths))    # one bright color per path

        # Plot all paths
        plt.figure(figsize=(10, 6))
        for i in range(n_paths):
            plt.plot(time_grid, paths[i], lw=1.0, alpha=0.65, color=colors[i])

        # Average path
        avg_path = paths.mean(axis=0)
        plt.plot(time_grid, avg_path, color="black", lw=2.0, label="Average Path")

        # Barrier
        plt.axhline(y=H, color="red", linestyle="--", lw=2, label=f"Barrier (H={H:.2f})")

        # Labels
        barrier_labels = {
            "down-out": "Down-and-Out",
            "down-in": "Down-and-In",
            "up-out": "Up-and-Out",
            "up-in": "Up-and-In"
        }
        plt.title(f"Simulated Paths for {barrier_labels.get(barrier_type, barrier_type)} {option_type.capitalize()} Option")
        plt.xlabel("Time (Years)")
        plt.ylabel("Stock Price")
        plt.legend()
        plt.show()
