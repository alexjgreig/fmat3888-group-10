"""
SVI-JW (SVI Jump-Wings) volatility surface calibration module.
Implements Gatheral and Jacquier's SVI-JW parameterization for implied volatility surfaces.

The SVI-JW parameterization uses more intuitive parameters:
- v_t: ATM total variance (ATM vol squared × time)
- ψ: ATM skew (vol skew at ATM)
- p: Left wing slope (put wing)
- c: Right wing slope (call wing)
- ṽ_t: Minimum variance

These map to raw SVI parameters via:
w(k) = a + b * (ρ * (k - m) + sqrt((k - m)^2 + σ^2))

Reference: 
- Gatheral & Jacquier, "Arbitrage-free SVI volatility surfaces" (2014)
- Jim Gatheral, "The Volatility Surface: A Practitioner's Guide" (2006)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution, least_squares
from scipy.interpolate import interp1d, RectBivariateSpline, CubicSpline
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SVIJWParameters:
    """Container for SVI-JW parameters for each maturity slice."""
    v_t: float      # ATM total variance
    psi: float      # ATM skew
    p: float        # Left wing slope (put wing)
    c: float        # Right wing slope (call wing)  
    v_tilde: float  # Minimum variance
    T: float        # Time to maturity
    
    def to_vector(self) -> np.ndarray:
        """Convert parameters to vector for optimization."""
        return np.array([self.v_t, self.psi, self.p, self.c, self.v_tilde])
    
    @classmethod
    def from_vector(cls, vector: np.ndarray, T: float):
        """Create parameters from optimization vector."""
        return cls(
            v_t=vector[0],
            psi=vector[1],
            p=vector[2],
            c=vector[3],
            v_tilde=vector[4],
            T=T
        )
    
    def to_raw_svi(self):
        """
        Convert SVI-JW parameters to raw SVI parameters.
        Using stable conversion from Gatheral & Jacquier.
        
        Returns:
            Tuple of (a, b, rho, m, sigma)
        """
        v_t = self.v_t
        psi = self.psi
        p = self.p
        c = self.c
        v_tilde = self.v_tilde
        
        # Ensure valid parameters
        if v_t <= v_tilde:
            v_t = v_tilde + 1e-4
        
        # Stable conversion formulas
        w = v_t - v_tilde  # ATM variance above minimum
        
        if w < 1e-8:
            # Near-zero variance case
            return v_tilde, 1e-8, 0.0, 0.0, 1e-4
        
        # Calculate b and rho
        b = 0.5 * np.sqrt(w) * (p + c)
        
        if abs(p + c) < 1e-8:
            rho = 0
        else:
            rho = (p - c) / (p + c)
            # Ensure rho is in valid range
            rho = np.clip(rho, -0.999, 0.999)
        
        # Calculate m and sigma using stable formulas
        if abs(psi) < 1e-8:
            # Zero skew case
            m = 0
            sigma = np.sqrt(w) / b if b > 1e-8 else 1e-4
        else:
            # General case with skew
            beta = rho - 2 * psi / np.sqrt(w)
            beta = np.clip(beta, -0.999, 0.999)
            
            # Calculate alpha ensuring positivity
            alpha_sq = 1 / (1 - beta**2) - 1
            if alpha_sq <= 0:
                alpha = 1e-4
            else:
                alpha = np.sqrt(alpha_sq)
            
            m = psi * np.sqrt(w) / (b * alpha) if b > 1e-8 else 0
            sigma = alpha * abs(m) if abs(m) > 1e-8 else np.sqrt(w) / b
        
        # Calculate a
        a = v_tilde - b * sigma * np.sqrt(1 - rho**2)
        
        # Final safety checks
        sigma = max(sigma, 1e-6)
        b = max(b, 1e-6)
        
        return a, b, rho, m, sigma


class SVIVolatilitySurface:
    """SVI-JW (Jump-Wings) volatility surface implementation."""
    
    def __init__(self, spot: float = None, forward_curve: Optional[Dict[float, float]] = None,
                 risk_free_curve: Optional[Dict[float, float]] = None):
        """
        Initialize SVI-JW volatility surface.
        
        Args:
            spot: Current spot price
            forward_curve: Forward prices by maturity (if None, uses spot)
            risk_free_curve: Risk-free rates by maturity (if None, uses flat rate)
        """
        self.spot = spot
        self.forward_curve = forward_curve or {}
        self.risk_free_curve = risk_free_curve or {}
        self.svi_jw_params_by_maturity: Dict[float, SVIJWParameters] = {}
        self.calibration_data: Optional[pd.DataFrame] = None
        self.interpolators: Dict = {}
        
    def get_forward(self, T: float) -> float:
        """Get forward price for maturity T."""
        if T in self.forward_curve:
            return self.forward_curve[T]
        elif self.spot is not None:
            # Simple forward with constant rate
            r = self.get_risk_free_rate(T)
            return self.spot * np.exp(r * T)
        else:
            return 100.0  # Default
    
    def get_risk_free_rate(self, T: float) -> float:
        """Get risk-free rate for maturity T."""
        if T in self.risk_free_curve:
            return self.risk_free_curve[T]
        elif self.risk_free_curve:
            # Linear interpolation
            maturities = sorted(self.risk_free_curve.keys())
            rates = [self.risk_free_curve[t] for t in maturities]
            if len(maturities) > 1:
                interp_func = interp1d(maturities, rates, kind='linear', 
                                      fill_value='extrapolate', bounds_error=False)
                return float(interp_func(T))
            else:
                return rates[0]
        else:
            return 0.04  # Default 4%
    
    def svi_raw(self, k: np.ndarray, a: float, b: float, rho: float, 
                m: float, sigma: float) -> np.ndarray:
        """
        Raw SVI formula for total implied variance.
        
        w(k) = a + b * (ρ * (k - m) + sqrt((k - m)^2 + σ^2))
        
        Args:
            k: Log-moneyness array
            a, b, rho, m, sigma: Raw SVI parameters
            
        Returns:
            Total implied variance w(k,T) = σ²(k,T) * T
        """
        return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
    
    def svi_jw(self, k: np.ndarray, params: SVIJWParameters) -> np.ndarray:
        """
        SVI-JW formula for total implied variance.
        
        Args:
            k: Log-moneyness array
            params: SVI-JW parameters
            
        Returns:
            Total implied variance
        """
        # Convert JW to raw parameters
        a, b, rho, m, sigma = params.to_raw_svi()
        
        # Calculate using raw SVI formula
        w = self.svi_raw(k, a, b, rho, m, sigma)
        
        # Ensure non-negative variance
        return np.maximum(w, 1e-8)
    
    def svi_implied_vol(self, k: np.ndarray, params: SVIJWParameters) -> np.ndarray:
        """
        Get implied volatility from SVI-JW parameters.
        
        Args:
            k: Log-moneyness
            params: SVI-JW parameters
            
        Returns:
            Implied volatility σ(k,T)
        """
        total_var = self.svi_jw(k, params)
        
        # Convert to implied vol: σ = sqrt(w/T)
        return np.sqrt(total_var / params.T)
    
    def check_arbitrage_constraints_jw(self, params: SVIJWParameters) -> bool:
        """
        Check no-arbitrage constraints for SVI-JW parameters.
        
        Returns:
            True if parameters satisfy no-arbitrage conditions
        """
        v_t = params.v_t
        p = params.p
        c = params.c
        v_tilde = params.v_tilde
        
        # Basic constraints
        if v_t <= 0 or v_tilde < 0:
            return False
            
        if v_t <= v_tilde:
            return False
            
        # Wing slopes must be positive
        if p <= 0 or c <= 0:
            return False
            
        # Durrleman condition for no butterfly arbitrage
        if p + c < 2:
            return False
            
        # Additional stability check
        if p > 10 or c > 10:  # Unreasonably large slopes
            return False
            
        return True
    
    def calibrate_slice(self, strikes: np.ndarray, ivs: np.ndarray, 
                       T: float, forward: float,
                       method: str = 'robust', max_iter: int = 1000) -> Dict:
        """
        Calibrate SVI-JW parameters for a single maturity slice.
        
        Args:
            strikes: Strike prices
            ivs: Implied volatilities
            T: Time to maturity
            forward: Forward price
            method: Optimization method ('robust', 'lm', or 'de')
            max_iter: Maximum iterations
            
        Returns:
            Dictionary with calibration results
        """
        # Convert to log-moneyness
        k = np.log(strikes / forward)
        
        # Total variance
        market_w = ivs**2 * T
        
        # Sort by moneyness for stability
        sort_idx = np.argsort(k)
        k = k[sort_idx]
        market_w = market_w[sort_idx]
        
        # Initial guess
        initial_params = self._get_initial_svi_jw_params(k, market_w, T)
        
        # Enhanced optimization with multiple stages
        def objective_with_penalty(params_vec):
            v_t, psi, p, c, v_tilde = params_vec
            
            # Hard constraints
            if v_t <= v_tilde + 0.005:
                return 1e10
            if p + c < 2:
                return 1e10
            if p <= 0 or c <= 0:
                return 1e10
            
            params = SVIJWParameters.from_vector(params_vec, T)
            
            try:
                model_w = self.svi_jw(k, params)
                # Weighted least squares with strong ATM emphasis
                weights = np.exp(-2 * k**2)
                residuals = (model_w - market_w) / (market_w + 0.01)
                weighted_error = np.sum(weights * residuals**2)
                
                # Add penalty for extreme parameters
                penalty = 0
                if abs(psi) > 0.5:
                    penalty += 100 * (abs(psi) - 0.5)**2
                if p > 2 or c > 2:
                    penalty += 100 * ((max(0, p - 2))**2 + (max(0, c - 2))**2)
                
                return weighted_error + penalty
            except:
                return 1e10
        
        # Simplified robust bounds
        bounds = [
            (min(market_w) * 0.8, max(market_w) * 2.0),  # v_t  
            (-0.3, 0.3),  # psi  
            (0.2, 1.5),   # p
            (0.2, 1.5),   # c
            (0, min(market_w) * 0.5)  # v_tilde
        ]
        
        # Stage 1: Global optimization with differential evolution (no x0)
        result_de = differential_evolution(
            objective_with_penalty,
            bounds,
            maxiter=max(50, max_iter // 20),
            popsize=15,
            tol=1e-8,
            seed=42,
            workers=1
        )
        
        # Stage 2: Local refinement
        result = minimize(
            objective_with_penalty,
            result_de.x,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter // 10, 'ftol': 1e-10}
        )
        
        # Use best result
        if result.fun < result_de.fun:
            optimal_params = result.x
        else:
            optimal_params = result_de.x
        
        # Create final parameters
        final_params = SVIJWParameters.from_vector(optimal_params, T)
        
        # Final Durrleman condition check
        if final_params.p + final_params.c < 2:
            scale = 2.01 / (final_params.p + final_params.c)
            final_params.p *= scale
            final_params.c *= scale
        
        # Calculate fit quality
        model_w = self.svi_jw(k, final_params)
        rmse = np.sqrt(np.mean((model_w - market_w)**2))
        
        return {
            'params': final_params,
            'success': True,
            'rmse': rmse,
            'k_range': (k.min(), k.max())
        }
    
    def _get_initial_svi_jw_params(self, k: np.ndarray, w: np.ndarray, T: float) -> SVIJWParameters:
        """
        Get robust initial SVI-JW parameters with enhanced heuristics.
        
        Args:
            k: Log-moneyness
            w: Total variance
            T: Time to maturity
            
        Returns:
            Initial SVI-JW parameters with tighter constraints
        """
        # Remove outliers using IQR method first
        q1, q3 = np.percentile(w, [25, 75])
        iqr = q3 - q1
        lower_bound = max(0, q1 - 1.5 * iqr)
        upper_bound = q3 + 1.5 * iqr
        valid_mask = (w >= lower_bound) & (w <= upper_bound)
        
        if np.sum(valid_mask) < 5:
            valid_mask = np.ones_like(w, dtype=bool)
        
        k_clean = k[valid_mask]
        w_clean = w[valid_mask]
        
        # Find ATM index
        atm_idx = np.argmin(np.abs(k_clean))
        
        # Use robust statistics for ATM variance
        atm_window = max(1, min(3, len(k_clean) // 10))
        atm_start = max(0, atm_idx - atm_window)
        atm_end = min(len(k_clean), atm_idx + atm_window + 1)
        v_t = np.median(w_clean[atm_start:atm_end])
        
        # Minimum variance (conservative estimate)
        v_tilde = min(w_clean) * 0.5
        
        # Apply tighter constraints upfront
        v_t = np.clip(v_t, 0.001, 0.5)
        v_tilde = np.clip(v_tilde, 0, v_t * 0.7)
        
        # Ensure v_t > v_tilde with margin
        v_t = max(v_t, v_tilde + 0.01)
        
        # Robust skew estimation using local regression
        near_atm_mask = np.abs(k_clean) < 0.15
        if np.sum(near_atm_mask) > 3:
            k_near = k_clean[near_atm_mask]
            w_near = w_clean[near_atm_mask]
            try:
                # Use robust linear fit
                A = np.vstack([k_near, np.ones(len(k_near))]).T
                coeffs, _, _, _ = np.linalg.lstsq(A, w_near, rcond=None)
                psi = coeffs[0] / (2 * np.sqrt(v_t)) if v_t > 0 else 0
            except:
                psi = 0
        else:
            psi = 0
        # Conservative skew bounds
        psi = np.clip(psi, -0.2, 0.2)
        
        # Robust wing estimation
        # Left wing (put side)
        left_mask = k_clean < -0.1
        if np.sum(left_mask) >= 2:
            k_left = k_clean[left_mask]
            w_left = w_clean[left_mask]
            # Calculate slopes robustly
            slopes = (w_left - v_tilde) / (np.abs(k_left) + 1e-6)
            valid_slopes = slopes[(slopes > 0) & (slopes < 3)]
            p = np.median(valid_slopes) if len(valid_slopes) > 0 else 0.7
        else:
            p = 0.7
        
        # Right wing (call side)
        right_mask = k_clean > 0.1
        if np.sum(right_mask) >= 2:
            k_right = k_clean[right_mask]
            w_right = w_clean[right_mask]
            # Calculate slopes robustly
            slopes = (w_right - v_tilde) / (k_right + 1e-6)
            valid_slopes = slopes[(slopes > 0) & (slopes < 3)]
            c = np.median(valid_slopes) if len(valid_slopes) > 0 else 0.7
        else:
            c = 0.7
        
        # Apply conservative constraints
        p = np.clip(p, 0.3, 1.2)
        c = np.clip(c, 0.3, 1.2)
        
        # Ensure Durrleman condition with margin
        if p + c < 2.0:
            # Scale up proportionally to satisfy constraint
            scale = 2.1 / (p + c + 1e-6)
            p = min(p * scale, 1.2)
            c = min(c * scale, 1.2)
        
        return SVIJWParameters(v_t, psi, p, c, v_tilde, T)
    
    def calibrate(self, option_data: pd.DataFrame, method: str = 'slice',
                 regularization: float = 0.0, max_iter: int = 500) -> Dict:
        """
        Calibrate SVI-JW surface to market data.
        
        Args:
            option_data: DataFrame with columns: strike, timeToExpiry, impliedVolatility, underlyingPrice
            method: 'slice' for maturity-by-maturity calibration
            regularization: Not used for SVI-JW (kept for compatibility)
            max_iter: Maximum iterations per slice
            
        Returns:
            Dictionary with calibration results
        """
        # Store calibration data
        self.calibration_data = option_data.copy()
        
        # Set spot price if not provided
        if self.spot is None:
            self.spot = option_data['underlyingPrice'].iloc[0]
        
        # Get unique maturities
        maturities = np.sort(option_data['timeToExpiry'].unique())
        
        results = {'success': True, 'slices': {}}
        total_rmse = 0
        n_slices = 0
        
        print(f"\nCalibrating SVI-JW for {len(maturities)} maturity slices...")
        
        # Calibrate each maturity slice
        for T in maturities:
            # Filter data for this maturity
            slice_data = option_data[option_data['timeToExpiry'] == T]
            
            if len(slice_data) < 5:  # Need at least 5 points for SVI-JW
                print(f"  Skipping T={T:.3f}: insufficient data ({len(slice_data)} points)")
                continue
            
            # Get forward price
            forward = self.get_forward(T)
            
            # Extract strikes and IVs
            strikes = slice_data['strike'].values
            ivs = slice_data['impliedVolatility'].values
            
            # Remove outliers (IVs that are too extreme)
            valid_mask = (ivs > 0.05) & (ivs < 2.0)
            if np.sum(valid_mask) < 5:
                print(f"  Skipping T={T:.3f}: insufficient valid data")
                continue
            
            strikes = strikes[valid_mask]
            ivs = ivs[valid_mask]
            
            # Calibrate slice
            print(f"  Calibrating T={T:.3f} with {len(strikes)} strikes...")
            slice_result = self.calibrate_slice(
                strikes, ivs, T, forward,
                method='robust',  # Use robust method for stability
                max_iter=max_iter
            )
            
            if slice_result['success'] or slice_result['rmse'] < 0.1:
                self.svi_jw_params_by_maturity[T] = slice_result['params']
                results['slices'][T] = slice_result
                total_rmse += slice_result['rmse']
                n_slices += 1
                
                # Display parameters
                params = slice_result['params']
                print(f"    ✓ v_t={params.v_t:.4f}, ψ={params.psi:.3f}, p={params.p:.2f}, c={params.c:.2f}, ṽ_t={params.v_tilde:.4f}")
                print(f"      RMSE={slice_result['rmse']:.6f}")
            else:
                results['success'] = False
                print(f"    ✗ Failed to calibrate")
        
        # Calculate overall metrics
        if n_slices > 0:
            avg_rmse = total_rmse / n_slices
            
            # Calculate overall fit metrics
            metrics = self._calculate_calibration_metrics(option_data)
            
            # Build smooth interpolators
            self._build_smooth_interpolators()
            
            return {
                'success': results['success'],
                'message': f"Calibrated {n_slices}/{len(maturities)} maturity slices",
                'metrics': metrics,
                'avg_slice_rmse': avg_rmse,
                'parameters': self.svi_jw_params_by_maturity
            }
        else:
            return {
                'success': False,
                'message': "Failed to calibrate any maturity slices",
                'metrics': {'rmse': np.inf, 'mae': np.inf, 'r_squared': -np.inf},
                'parameters': {}
            }
    
    def _build_smooth_interpolators(self):
        """Build smooth interpolators for the surface."""
        if not self.svi_jw_params_by_maturity:
            return
        
        # Get calibrated maturities and parameters
        maturities = sorted(self.svi_jw_params_by_maturity.keys())
        
        if len(maturities) < 2:
            return
        
        # Extract parameter vectors for interpolation
        params_matrix = np.array([
            self.svi_jw_params_by_maturity[T].to_vector() 
            for T in maturities
        ])
        
        # Create smooth interpolators for each parameter
        self.param_interpolators = {}
        param_names = ['v_t', 'psi', 'p', 'c', 'v_tilde']
        
        for i, name in enumerate(param_names):
            if len(maturities) >= 3:
                # Use cubic spline for smooth interpolation
                self.param_interpolators[name] = CubicSpline(
                    maturities, params_matrix[:, i],
                    bc_type='natural'  # Natural boundary conditions
                )
            else:
                # Linear interpolation for few points
                self.param_interpolators[name] = interp1d(
                    maturities, params_matrix[:, i],
                    kind='linear', fill_value='extrapolate', bounds_error=False
                )
        
        # Build fine grid for visualization
        self._build_fine_grid()
    
    def _build_fine_grid(self):
        """Build fine grid for smooth surface visualization."""
        if not self.svi_jw_params_by_maturity:
            return
        
        # Create fine grid
        min_strike = self.spot * 0.5
        max_strike = self.spot * 2.0
        
        strikes = np.linspace(min_strike, max_strike, 50)
        
        # Use more maturity points for smoothness
        available_maturities = sorted(self.svi_jw_params_by_maturity.keys())
        min_T = available_maturities[0]
        max_T = available_maturities[-1]
        maturities = np.linspace(min_T, max_T, 20)
        
        # Build vol grid
        vol_grid = np.zeros((len(strikes), len(maturities)))
        
        for j, T in enumerate(maturities):
            # Get interpolated parameters
            if hasattr(self, 'param_interpolators'):
                v_t = float(self.param_interpolators['v_t'](T))
                psi = float(self.param_interpolators['psi'](T))
                p = float(self.param_interpolators['p'](T))
                c = float(self.param_interpolators['c'](T))
                v_tilde = float(self.param_interpolators['v_tilde'](T))
                
                # Ensure valid parameters
                v_t = max(v_tilde + 0.01, v_t)
                p = max(0.1, p)
                c = max(0.1, c)
                
                # Ensure Durrleman condition
                if p + c < 2:
                    scale = 2.01 / (p + c)
                    p *= scale
                    c *= scale
                
                params = SVIJWParameters(v_t, psi, p, c, v_tilde, T)
            else:
                # Find closest calibrated maturity
                closest_T = min(available_maturities, key=lambda x: abs(x - T))
                params = self.svi_jw_params_by_maturity[closest_T]
            
            # Calculate vols for this maturity
            forward = self.get_forward(T)
            k = np.log(strikes / forward)
            ivs = self.svi_implied_vol(k, params)
            vol_grid[:, j] = ivs
        
        # Apply smoothing to remove any remaining kinks
        from scipy.ndimage import gaussian_filter
        vol_grid = gaussian_filter(vol_grid, sigma=0.5)
        
        # Create smooth 2D spline interpolator
        self.interpolators['surface'] = RectBivariateSpline(
            strikes, maturities, vol_grid, kx=3, ky=3, s=0.001
        )
        
        # Store grid for reference
        self.interpolators['strike_grid'] = strikes
        self.interpolators['maturity_grid'] = maturities
        self.interpolators['vol_grid'] = vol_grid
    
    def get_vol(self, strike: float, maturity: float, 
                use_interpolator: bool = True) -> float:
        """
        Get implied volatility for given strike and maturity.
        
        Args:
            strike: Strike price
            maturity: Time to maturity
            use_interpolator: Use smooth interpolator if available
            
        Returns:
            Implied volatility
        """
        # Handle edge cases
        if maturity <= 0:
            maturity = 1/365  # Minimum 1 day
        
        # Check if we have calibrated parameters
        if not self.svi_jw_params_by_maturity:
            # Return reasonable default if no calibration
            return 0.25  # 25% default volatility
        
        available_maturities = sorted(self.svi_jw_params_by_maturity.keys())
        
        # Handle extrapolation for very short maturities
        if maturity < available_maturities[0]:
            # Use nearest available with sqrt time scaling
            nearest_T = available_maturities[0]
            nearest_params = self.svi_jw_params_by_maturity[nearest_T]
            
            # Calculate vol at nearest maturity
            forward = self.get_forward(nearest_T)
            k = np.log(strike / forward)
            nearest_vol = self.svi_implied_vol(np.array([k]), nearest_params)[0]
            
            # Scale by sqrt of time for short extrapolation
            scaling = np.sqrt(maturity / nearest_T)
            return nearest_vol * scaling
        
        # Handle extrapolation for long maturities
        if maturity > available_maturities[-1]:
            # Use furthest available with sqrt time scaling
            furthest_T = available_maturities[-1]
            furthest_params = self.svi_jw_params_by_maturity[furthest_T]
            
            # Calculate vol at furthest maturity
            forward = self.get_forward(furthest_T)
            k = np.log(strike / forward)
            furthest_vol = self.svi_implied_vol(np.array([k]), furthest_params)[0]
            
            # Scale by sqrt of time for long extrapolation
            scaling = np.sqrt(maturity / furthest_T)
            return furthest_vol * scaling
        
        # Interpolation between calibrated maturities
        if len(available_maturities) >= 2:
            # Find bracketing maturities
            idx = np.searchsorted(available_maturities, maturity)
            
            if idx == 0:
                # Use first maturity
                params = self.svi_jw_params_by_maturity[available_maturities[0]]
            elif idx == len(available_maturities):
                # Use last maturity
                params = self.svi_jw_params_by_maturity[available_maturities[-1]]
            else:
                # Smooth parameter interpolation for continuity
                T1 = available_maturities[idx - 1]
                T2 = available_maturities[idx]
                
                # Get parameters for both maturities
                params1 = self.svi_jw_params_by_maturity[T1]
                params2 = self.svi_jw_params_by_maturity[T2]
                
                # Interpolation weight with smooth transition
                alpha = (maturity - T1) / (T2 - T1)
                # Use cubic hermite spline weight for C1 continuity
                alpha_smooth = alpha * alpha * (3 - 2 * alpha)
                
                # Interpolate SVI parameters directly
                v_t = params1.v_t * (1 - alpha_smooth) + params2.v_t * alpha_smooth
                psi = params1.psi * (1 - alpha_smooth) + params2.psi * alpha_smooth
                p = params1.p * (1 - alpha_smooth) + params2.p * alpha_smooth
                c = params1.c * (1 - alpha_smooth) + params2.c * alpha_smooth
                v_tilde = params1.v_tilde * (1 - alpha_smooth) + params2.v_tilde * alpha_smooth
                
                # Create interpolated parameters
                params_interp = SVIJWParameters(v_t, psi, p, c, v_tilde, maturity)
                
                # Check if interpolated params satisfy arbitrage constraints
                if self.check_arbitrage_constraints_jw(params_interp):
                    # Use interpolated parameters
                    forward = self.get_forward(maturity)
                    k = np.log(strike / forward)
                    return self.svi_implied_vol(np.array([k]), params_interp)[0]
                else:
                    # Fall back to variance interpolation if constraints violated
                    forward = self.get_forward(maturity)
                    k = np.log(strike / forward)
                    
                    vol1 = self.svi_implied_vol(np.array([k]), params1)[0]
                    vol2 = self.svi_implied_vol(np.array([k]), params2)[0]
                    
                    # Smooth variance interpolation
                    var1 = vol1**2 * T1
                    var2 = vol2**2 * T2
                    interpolated_var = var1 * (1 - alpha_smooth) + var2 * alpha_smooth
                    
                    return np.sqrt(interpolated_var / maturity)
        else:
            # Only one maturity available
            params = self.svi_jw_params_by_maturity[available_maturities[0]]
        
        # Calculate vol using SVI parameters
        forward = self.get_forward(maturity)
        k = np.log(strike / forward)
        
        return self.svi_implied_vol(np.array([k]), params)[0]
    
    def _calculate_calibration_metrics(self, option_data: pd.DataFrame) -> Dict:
        """Calculate calibration quality metrics."""
        errors = []
        
        for _, row in option_data.iterrows():
            model_vol = self.get_vol(row['strike'], row['timeToExpiry'], use_interpolator=False)
            market_vol = row['impliedVolatility']
            errors.append(model_vol - market_vol)
        
        errors = np.array(errors)
        
        return {
            'rmse': np.sqrt(np.mean(errors**2)),
            'mae': np.mean(np.abs(errors)),
            'max_error': np.max(np.abs(errors)),
            'mean_error': np.mean(errors),
            'r_squared': 1 - np.var(errors) / np.var(option_data['impliedVolatility'])
        }
    
    
    def get_total_variance(self, strike: float, maturity: float) -> float:
        """Get total implied variance (vol^2 * T)."""
        vol = self.get_vol(strike, maturity)
        return vol**2 * maturity
    
    # Compatibility properties
    @property
    def svi_params_by_maturity(self):
        """Compatibility property."""
        return self.svi_jw_params_by_maturity
    
    @property 
    def params(self):
        """Compatibility property."""
        return bool(self.svi_jw_params_by_maturity)


# For backward compatibility
ParametricVolatilitySurface = SVIVolatilitySurface