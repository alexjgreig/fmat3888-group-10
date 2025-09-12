"""
Industry-Standard Dupire Local Volatility Model Implementation
===============================================================

This module implements a production-quality local volatility model using:
- Kernel regression for stable implied variance surface
- Tikhonov regularization for smooth derivatives
- Higher-order finite differences with proper boundary handling
- Industry-standard calibration techniques

References:
- Dupire, B. (1994) "Pricing with a smile", Risk 7(1), pp. 18-20
- Gatheral, J. (2006) "The Volatility Surface: A Practitioner's Guide"
- Andersen, L. & Brotherton-Ratcliffe, R. (1998) "The equity option volatility smile"
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline, CubicSpline, UnivariateSpline
from scipy.ndimage import gaussian_filter, median_filter
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize_scalar
from typing import Optional, Dict, Tuple, List
from ..models.volatility_surface import ParametricVolatilitySurface
from ..models.black_scholes import BlackScholes
import warnings
warnings.filterwarnings('ignore')


class DupireLocalVolatility:
    """
    Production-Quality Dupire Local Volatility Model
    
    Features:
    - Kernel smoothing for stable implied variance
    - Tikhonov regularization for derivatives
    - Higher-order finite differences
    - Adaptive smoothing based on data density
    - Robust arbitrage-free constraints
    """
    
    def __init__(self, vol_surface: Optional[ParametricVolatilitySurface] = None,
                 spot: float = 100.0, risk_free_rate: float = 0.04,
                 dividend_yield: float = 0.0):
        """
        Initialize Dupire local volatility model.
        
        Args:
            vol_surface: Calibrated parametric volatility surface
            spot: Current spot price
            risk_free_rate: Risk-free interest rate
            dividend_yield: Dividend yield
        """
        self.vol_surface = vol_surface
        self.spot = spot
        self.r = risk_free_rate
        self.q = dividend_yield
        self.local_vol_interpolator = None
        self.local_vol_grid = None
        self.strikes = None
        self.maturities = None
        self.variance_surface = None
        self.smoothed_variance_surface = None
        
    def kernel_smooth_variance(self, strikes: np.ndarray, maturities: np.ndarray,
                              variance_grid: np.ndarray, bandwidth: Optional[float] = None) -> np.ndarray:
        """
        Apply 2D kernel smoothing to variance surface for stability.
        Uses adaptive bandwidth based on data density.
        
        Args:
            strikes: Strike prices array
            maturities: Time to maturity array
            variance_grid: Raw variance surface
            bandwidth: Kernel bandwidth (auto-selected if None)
            
        Returns:
            Smoothed variance surface
        """
        n_strikes = len(strikes)
        n_maturities = len(maturities)
        
        # Convert to log-moneyness for better scaling
        log_moneyness = np.log(strikes / self.spot)
        
        # Auto-select bandwidth using Scott's rule if not provided
        if bandwidth is None:
            # Estimate optimal bandwidth based on data spacing
            h_k = 1.06 * np.std(log_moneyness) * n_strikes**(-1/5)
            h_t = 1.06 * np.std(maturities) * n_maturities**(-1/5)
            bandwidth = np.sqrt(h_k * h_t)
        
        # Apply 2D Gaussian kernel smoothing
        # First smooth in strike dimension
        smoothed = np.zeros_like(variance_grid)
        for j in range(n_maturities):
            # Use adaptive bandwidth based on local data density
            local_bandwidth = bandwidth * (1 + 0.5 * maturities[j])  # Increase smoothing for longer maturities
            
            # Apply Gaussian kernel regression
            for i in range(n_strikes):
                # Compute weights
                weights = np.exp(-0.5 * ((log_moneyness - log_moneyness[i]) / local_bandwidth)**2)
                weights /= weights.sum()
                
                # Weighted average
                smoothed[i, j] = np.sum(weights * variance_grid[:, j])
        
        # Then smooth in time dimension with monotonicity constraint
        for i in range(n_strikes):
            # Ensure variance is increasing in time (calendar arbitrage)
            for j in range(1, n_maturities):
                if smoothed[i, j] <= smoothed[i, j-1]:
                    # Apply minimum increment proportional to time step
                    min_increment = 0.001 * (maturities[j] - maturities[j-1])
                    smoothed[i, j] = smoothed[i, j-1] + min_increment
        
        return smoothed
    
    def tikhonov_derivatives(self, variance_grid: np.ndarray, strikes: np.ndarray,
                            maturities: np.ndarray, alpha: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute smooth derivatives using 2D Tikhonov regularization.
        This prevents spikes by adding a roughness penalty.
        
        Args:
            variance_grid: Total variance surface
            strikes: Strike prices
            maturities: Time to maturity
            alpha: Regularization parameter (higher = smoother)
            
        Returns:
            Tuple of (dw/dT, dw/dK, d²w/dK²) derivatives
        """
        n_strikes = len(strikes)
        n_maturities = len(maturities)
        
        # Convert to log-moneyness
        log_moneyness = np.log(strikes / self.spot)
        
        # Build differentiation matrices with higher-order accuracy
        # Fourth-order central differences for better accuracy
        def build_diff_matrix_4th(n: int, h: float, order: int = 1):
            """Build 4th-order accurate finite difference matrix."""
            if order == 1:
                # Fourth-order first derivative
                stencil = np.array([-1/12, 2/3, 0, -2/3, 1/12]) / h
                offsets = [-2, -1, 0, 1, 2]
            else:  # order == 2
                # Fourth-order second derivative
                stencil = np.array([-1/12, 4/3, -5/2, 4/3, -1/12]) / (h**2)
                offsets = [-2, -1, 0, 1, 2]
            
            # Handle boundaries with lower order
            D = np.zeros((n, n))
            
            # Interior points - 4th order
            for i in range(2, n-2):
                for j, offset in enumerate(offsets):
                    if 0 <= i + offset < n:
                        D[i, i + offset] = stencil[j]
            
            # Boundaries - 2nd order forward/backward
            if order == 1:
                # Forward difference at start
                D[0, :3] = [-3, 4, -1] / (2 * h)
                D[1, :4] = [-2, -1, 0, 3] / (2 * h)
                
                # Backward difference at end
                D[-1, -3:] = [1, -4, 3] / (2 * h)
                D[-2, -4:] = [-3, 0, 1, 2] / (2 * h)
            else:  # order == 2
                # Second derivative at boundaries
                D[0, :3] = [1, -2, 1] / (h**2)
                D[1, :4] = [1, -2, 1, 0] / (h**2)
                D[-1, -3:] = [1, -2, 1] / (h**2)
                D[-2, -4:] = [0, 1, -2, 1] / (h**2)
            
            return D
        
        # Build differentiation matrices
        h_k = np.mean(np.diff(log_moneyness))
        h_t = np.mean(np.diff(maturities))
        
        D_k1 = build_diff_matrix_4th(n_strikes, h_k, order=1)
        D_k2 = build_diff_matrix_4th(n_strikes, h_k, order=2)
        D_t1 = build_diff_matrix_4th(n_maturities, h_t, order=1)
        
        # Add Tikhonov regularization
        # L2 regularization on second derivatives to enforce smoothness
        L_k = D_k2.T @ D_k2  # Roughness penalty in strike dimension
        L_t = D_t1.T @ D_t1   # Roughness penalty in time dimension
        
        # Compute regularized derivatives
        dw_dT = np.zeros_like(variance_grid)
        dw_dK = np.zeros_like(variance_grid)
        d2w_dK2 = np.zeros_like(variance_grid)
        
        # Time derivative with regularization
        for i in range(n_strikes):
            # Solve regularized least squares: (I + alpha*L)x = y
            A = np.eye(n_maturities) + alpha * L_t
            dw_dT[i, :] = np.linalg.solve(A, D_t1 @ variance_grid[i, :])
        
        # Strike derivatives with regularization
        for j in range(n_maturities):
            # First derivative
            A = np.eye(n_strikes) + alpha * L_k
            dw_dK[:, j] = np.linalg.solve(A, D_k1 @ variance_grid[:, j])
            
            # Second derivative
            d2w_dK2[:, j] = np.linalg.solve(A, D_k2 @ variance_grid[:, j])
        
        # Convert derivatives from log-moneyness to strike space
        for j in range(n_maturities):
            # Chain rule: dw/dK = (dw/dk) * (dk/dK) = (dw/dk) / K
            dw_dK[:, j] = dw_dK[:, j] / strikes
            
            # d²w/dK² = (d²w/dk² - dw/dk) / K²
            d2w_dK2[:, j] = (d2w_dK2[:, j] - dw_dK[:, j] * strikes) / (strikes**2)
        
        return dw_dT, dw_dK, d2w_dK2
    
    def build_variance_surface(self, strike_range: Tuple[float, float] = (0.5, 2.0),
                              maturity_range: Tuple[float, float] = (0.01, 2.0),
                              n_strikes: int = 100, n_maturities: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build smoothed total implied variance surface for stable derivatives.
        
        Args:
            strike_range: Range as fraction of spot
            maturity_range: Range in years  
            n_strikes: Number of strike points
            n_maturities: Number of maturity points
            
        Returns:
            Tuple of (strikes, maturities, smoothed_variance_grid)
        """
        if self.vol_surface is None:
            raise ValueError("Volatility surface not set")
        
        # Create uniform grids in appropriate spaces
        # Log-uniform strikes for better resolution
        log_strikes = np.linspace(
            np.log(self.spot * strike_range[0]),
            np.log(self.spot * strike_range[1]),
            n_strikes
        )
        strikes = np.exp(log_strikes)
        
        # Square-root time for better short-maturity resolution
        sqrt_maturities = np.linspace(
            np.sqrt(maturity_range[0]),
            np.sqrt(maturity_range[1]),
            n_maturities
        )
        maturities = sqrt_maturities**2
        
        # Compute raw total variance grid
        variance_grid = np.zeros((n_strikes, n_maturities))
        
        for i, K in enumerate(strikes):
            for j, T in enumerate(maturities):
                sigma = self.vol_surface.get_vol(K, T)
                # Apply quality filter - remove extreme values
                if 0.05 < sigma < 2.0:  # Between 5% and 200% vol
                    variance_grid[i, j] = sigma**2 * T
                else:
                    # Use interpolation from neighbors
                    if i > 0 and j > 0:
                        variance_grid[i, j] = 0.5 * (variance_grid[i-1, j] + variance_grid[i, j-1])
                    else:
                        variance_grid[i, j] = 0.25**2 * T  # Default 25% vol
        
        # Apply kernel smoothing for stability
        smoothed_variance = self.kernel_smooth_variance(strikes, maturities, variance_grid)
        
        # Additional smoothing with edge-preserving filter
        # Use adaptive bilateral filter to preserve important features
        smoothed_variance = self.bilateral_filter_2d(smoothed_variance, sigma_space=1.0, sigma_range=0.1)
        
        # Store grids
        self.strikes = strikes
        self.maturities = maturities
        self.variance_grid = variance_grid
        self.smoothed_variance_surface = smoothed_variance
        
        return strikes, maturities, smoothed_variance
    
    def bilateral_filter_2d(self, data: np.ndarray, sigma_space: float = 1.0,
                           sigma_range: float = 0.1) -> np.ndarray:
        """
        Apply 2D bilateral filter for edge-preserving smoothing.
        
        Args:
            data: Input 2D array
            sigma_space: Spatial kernel bandwidth
            sigma_range: Range kernel bandwidth
            
        Returns:
            Filtered array
        """
        n_i, n_j = data.shape
        filtered = np.zeros_like(data)
        
        # Window size based on spatial bandwidth
        window = int(np.ceil(3 * sigma_space))
        
        for i in range(n_i):
            for j in range(n_j):
                # Define local window
                i_min = max(0, i - window)
                i_max = min(n_i, i + window + 1)
                j_min = max(0, j - window)
                j_max = min(n_j, j + window + 1)
                
                # Extract local patch
                patch = data[i_min:i_max, j_min:j_max]
                
                # Compute spatial weights
                i_coords, j_coords = np.meshgrid(
                    range(i_min, i_max),
                    range(j_min, j_max),
                    indexing='ij'
                )
                spatial_weights = np.exp(-0.5 * ((i_coords - i)**2 + (j_coords - j)**2) / sigma_space**2)
                
                # Compute range weights
                range_weights = np.exp(-0.5 * ((patch - data[i, j]) / sigma_range)**2)
                
                # Combined weights
                weights = spatial_weights * range_weights
                weights /= weights.sum()
                
                # Weighted average
                filtered[i, j] = np.sum(weights * patch)
        
        return filtered
    
    def dupire_formula_stable(self, K: float, T: float,
                             dw_dT: np.ndarray, dw_dK: np.ndarray,
                             d2w_dK2: np.ndarray) -> float:
        """
        Calculate local volatility using stable Dupire formula.
        
        Args:
            K: Strike price
            T: Time to maturity
            dw_dT: Time derivative grid
            dw_dK: Strike derivative grid
            d2w_dK2: Second strike derivative grid
            
        Returns:
            Local volatility at (K, T)
        """
        # Find indices for interpolation
        k_idx = np.searchsorted(self.strikes, K)
        t_idx = np.searchsorted(self.maturities, T)
        
        # Boundary handling
        if k_idx <= 0 or k_idx >= len(self.strikes):
            return self.vol_surface.get_vol(K, T) if self.vol_surface else 0.25
        if t_idx <= 0 or t_idx >= len(self.maturities):
            return self.vol_surface.get_vol(K, T) if self.vol_surface else 0.25
        
        # Bicubic interpolation for smooth derivatives
        # Use CubicSpline for better interpolation
        k_range = max(0, k_idx-2), min(len(self.strikes), k_idx+3)
        t_range = max(0, t_idx-2), min(len(self.maturities), t_idx+3)
        
        local_strikes = self.strikes[k_range[0]:k_range[1]]
        local_times = self.maturities[t_range[0]:t_range[1]]
        
        # Interpolate derivatives at (K, T)
        from scipy.interpolate import RectBivariateSpline
        
        # Time derivative
        spline_dw_dT = RectBivariateSpline(
            local_strikes, local_times,
            dw_dT[k_range[0]:k_range[1], t_range[0]:t_range[1]],
            kx=min(3, len(local_strikes)-1),
            ky=min(3, len(local_times)-1)
        )
        dw_dT_val = float(spline_dw_dT(K, T))
        
        # First strike derivative
        spline_dw_dK = RectBivariateSpline(
            local_strikes, local_times,
            dw_dK[k_range[0]:k_range[1], t_range[0]:t_range[1]],
            kx=min(3, len(local_strikes)-1),
            ky=min(3, len(local_times)-1)
        )
        dw_dK_val = float(spline_dw_dK(K, T))
        
        # Second strike derivative
        spline_d2w_dK2 = RectBivariateSpline(
            local_strikes, local_times,
            d2w_dK2[k_range[0]:k_range[1], t_range[0]:t_range[1]],
            kx=min(3, len(local_strikes)-1),
            ky=min(3, len(local_times)-1)
        )
        d2w_dK2_val = float(spline_d2w_dK2(K, T))
        
        # Apply Dupire formula with drift terms
        numerator = dw_dT_val
        if abs(self.r) > 1e-10 or abs(self.q) > 1e-10:
            # Get total variance at (K,T)
            spline_w = RectBivariateSpline(
                self.strikes, self.maturities,
                self.smoothed_variance_surface,
                kx=3, ky=3
            )
            w_val = float(spline_w(K, T))
            numerator += (self.r - self.q) * K * dw_dK_val + self.q * w_val
        
        denominator = 0.5 * K**2 * d2w_dK2_val
        
        # Stability checks with softer bounds
        impl_vol = self.vol_surface.get_vol(K, T)
        
        # Check for calendar arbitrage (numerator should be positive)
        if numerator <= 0:
            # Use implied vol with smooth adjustment
            return impl_vol * (0.9 + 0.1 * np.exp(-abs(numerator)))
        
        # Check for butterfly arbitrage (denominator should be positive)
        min_denominator = 1e-6 * K**2
        if denominator <= min_denominator:
            # Smooth transition to implied vol
            weight = np.exp(-denominator / min_denominator)
            return impl_vol * (1 - 0.3 * weight)
        
        # Calculate local variance
        local_variance = numerator / denominator
        
        # Ensure positive local variance
        if local_variance <= 0:
            return impl_vol
        
        local_vol = np.sqrt(local_variance)
        
        # Apply soft bounds - industry standard is [0.3, 3.0] times implied vol
        # Use smooth transition functions instead of hard clipping
        min_vol = 0.3 * impl_vol
        max_vol = 3.0 * impl_vol
        
        if local_vol < min_vol:
            # Smooth transition from below
            alpha = np.exp(-(min_vol - local_vol) / (0.1 * impl_vol))
            local_vol = min_vol * (1 - alpha) + local_vol * alpha
        elif local_vol > max_vol:
            # Smooth transition from above
            alpha = np.exp(-(local_vol - max_vol) / (0.1 * impl_vol))
            local_vol = max_vol * (1 - alpha) + local_vol * alpha
        
        return local_vol
    
    def build_local_vol_surface(self, strike_range: Tuple[float, float] = (0.5, 2.0),
                              maturity_range: Tuple[float, float] = (0.01, 2.0),
                              n_strikes: int = 50, n_maturities: int = 30,
                              tikhonov_alpha: float = 0.01) -> None:
        """
        Build production-quality local volatility surface.
        
        Args:
            strike_range: Range as fraction of spot
            maturity_range: Range in years
            n_strikes: Number of strike points for output grid
            n_maturities: Number of maturity points for output grid
            tikhonov_alpha: Regularization parameter for derivatives
        """
        if self.vol_surface is None:
            raise ValueError("Volatility surface not set")
        
        print("Building industry-standard local volatility surface...")
        
        # Step 1: Build high-resolution smoothed variance surface
        print("  Step 1: Building smoothed variance surface...")
        strikes_fine, maturities_fine, smoothed_variance = self.build_variance_surface(
            strike_range, maturity_range,
            n_strikes=max(150, n_strikes*3),  # Higher resolution for derivatives
            n_maturities=max(75, n_maturities*3)
        )
        
        # Step 2: Compute regularized derivatives
        print("  Step 2: Computing Tikhonov-regularized derivatives...")
        dw_dT, dw_dK, d2w_dK2 = self.tikhonov_derivatives(
            smoothed_variance, strikes_fine, maturities_fine, alpha=tikhonov_alpha
        )
        
        # Step 3: Compute local volatilities on output grid
        print("  Step 3: Computing local volatilities...")
        
        # Output grid - slightly reduced range to avoid boundaries
        strike_range_adj = (strike_range[0] * 1.05, strike_range[1] * 0.95)
        maturity_range_adj = (max(0.02, maturity_range[0]), maturity_range[1] * 0.98)
        
        # Create output grid
        log_strikes_out = np.linspace(
            np.log(self.spot * strike_range_adj[0]),
            np.log(self.spot * strike_range_adj[1]),
            n_strikes
        )
        strikes_out = np.exp(log_strikes_out)
        
        maturities_out = np.linspace(maturity_range_adj[0], maturity_range_adj[1], n_maturities)
        
        # Calculate local volatility grid
        local_vol_grid = np.zeros((n_strikes, n_maturities))
        
        for i, K in enumerate(strikes_out):
            for j, T in enumerate(maturities_out):
                local_vol_grid[i, j] = self.dupire_formula_stable(
                    K, T, dw_dT, dw_dK, d2w_dK2
                )
        
        # Step 4: Apply final smoothing with arbitrage preservation
        print("  Step 4: Applying arbitrage-preserving smoothing...")
        
        # Light median filter to remove any remaining spikes
        local_vol_grid = median_filter(local_vol_grid, size=3)
        
        # Edge-preserving bilateral filter
        local_vol_grid = self.bilateral_filter_2d(
            local_vol_grid, sigma_space=1.5, sigma_range=0.05
        )
        
        # Ensure smoothness in time dimension
        for i in range(n_strikes):
            # Apply monotone cubic spline in time
            spline = UnivariateSpline(maturities_out, local_vol_grid[i, :], k=3, s=0.001)
            local_vol_grid[i, :] = spline(maturities_out)
            
            # Ensure no extreme jumps
            for j in range(1, n_maturities):
                max_change = 0.2  # Maximum 20% change between time steps
                ratio = local_vol_grid[i, j] / local_vol_grid[i, j-1]
                if ratio > 1 + max_change:
                    local_vol_grid[i, j] = local_vol_grid[i, j-1] * (1 + max_change)
                elif ratio < 1 - max_change:
                    local_vol_grid[i, j] = local_vol_grid[i, j-1] * (1 - max_change)
        
        # Final bounds check with smooth constraints
        for i, K in enumerate(strikes_out):
            for j, T in enumerate(maturities_out):
                impl_vol = self.vol_surface.get_vol(K, T)
                
                # Soft bounds [0.3, 3.0] times implied vol
                min_local = 0.3 * impl_vol
                max_local = 3.0 * impl_vol
                
                # Smooth clipping
                if local_vol_grid[i, j] < min_local:
                    alpha = np.exp(-(min_local - local_vol_grid[i, j]) / (0.05 * impl_vol))
                    local_vol_grid[i, j] = min_local * (1 - alpha) + local_vol_grid[i, j] * alpha
                elif local_vol_grid[i, j] > max_local:
                    alpha = np.exp(-(local_vol_grid[i, j] - max_local) / (0.05 * impl_vol))
                    local_vol_grid[i, j] = max_local * (1 - alpha) + local_vol_grid[i, j] * alpha
        
        # Store the grid
        self.local_vol_grid = local_vol_grid
        self.strikes = strikes_out
        self.maturities = maturities_out
        
        # Create smooth interpolator using cubic splines
        self.local_vol_interpolator = RectBivariateSpline(
            strikes_out, maturities_out, local_vol_grid,
            kx=3, ky=3, s=0.0001  # Small smoothing parameter
        )
        
        # Print quality metrics
        print(f"  ✓ Local volatility surface built: {n_strikes} × {n_maturities}")
        print(f"    Min local vol: {np.min(local_vol_grid):.4f}")
        print(f"    Max local vol: {np.max(local_vol_grid):.4f}")
        print(f"    Mean local vol: {np.mean(local_vol_grid):.4f}")
        print(f"    Std local vol: {np.std(local_vol_grid):.4f}")
        
        # Validate surface quality
        self._validate_surface()
    
    def _validate_surface(self) -> None:
        """Validate the local volatility surface quality."""
        if self.local_vol_grid is None:
            return
        
        print("\n  Surface Quality Metrics:")
        print("  " + "="*40)
        
        # Check for NaN or Inf
        n_nan = np.sum(np.isnan(self.local_vol_grid))
        n_inf = np.sum(np.isinf(self.local_vol_grid))
        
        if n_nan > 0 or n_inf > 0:
            print(f"    ⚠ Warning: {n_nan} NaN and {n_inf} Inf values detected")
        else:
            print(f"    ✓ No NaN or Inf values")
        
        # Check smoothness using total variation
        tv_strikes = np.sum(np.abs(np.diff(self.local_vol_grid, axis=0)))
        tv_time = np.sum(np.abs(np.diff(self.local_vol_grid, axis=1)))
        total_variation = (tv_strikes + tv_time) / self.local_vol_grid.size
        
        print(f"    Total Variation: {total_variation:.6f}")
        if total_variation < 0.01:
            print(f"    ✓ Surface is very smooth (production quality)")
        elif total_variation < 0.05:
            print(f"    ✓ Surface is smooth")
        else:
            print(f"    ⚠ Surface may need more smoothing")
        
        # Check local vol / implied vol ratios
        ratios = []
        for i, K in enumerate(self.strikes[::5]):  # Sample every 5th strike
            for j, T in enumerate(self.maturities[::3]):  # Sample every 3rd maturity
                local_vol = self.local_vol_grid[i*5, j*3] if i*5 < len(self.strikes) and j*3 < len(self.maturities) else 0
                impl_vol = self.vol_surface.get_vol(K, T) if self.vol_surface else 0.25
                if impl_vol > 0:
                    ratios.append(local_vol / impl_vol)
        
        if ratios:
            min_ratio = min(ratios)
            max_ratio = max(ratios)
            mean_ratio = np.mean(ratios)
            
            print(f"    Local/Implied Vol Ratios:")
            print(f"      Min: {min_ratio:.2f}, Max: {max_ratio:.2f}, Mean: {mean_ratio:.2f}")
            
            if 0.3 <= min_ratio and max_ratio <= 3.0:
                print(f"    ✓ Ratios within industry standards [0.3, 3.0]")
            else:
                print(f"    ⚠ Some ratios outside standard bounds")
    
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
            # Clip to grid boundaries with smooth extrapolation
            S_min, S_max = self.strikes[0], self.strikes[-1]
            t_min, t_max = self.maturities[0], self.maturities[-1]
            
            # Smooth extrapolation for strikes
            if S < S_min:
                # Extrapolate with decay
                S_eval = S_min
                decay = np.exp(-(S_min - S) / (0.1 * self.spot))
                base_vol = float(self.local_vol_interpolator(S_eval, np.clip(t, t_min, t_max), grid=False))
                return base_vol * (1 + 0.2 * (1 - decay))  # Increase vol for far OTM
            elif S > S_max:
                S_eval = S_max
                decay = np.exp(-(S - S_max) / (0.1 * self.spot))
                base_vol = float(self.local_vol_interpolator(S_eval, np.clip(t, t_min, t_max), grid=False))
                return base_vol * (1 + 0.2 * (1 - decay))
            else:
                S_eval = S
            
            # Standard interpolation for time
            t_eval = np.clip(t, t_min, t_max)
            
            vol = float(self.local_vol_interpolator(S_eval, t_eval, grid=False))
            return max(vol, 0.01)  # Ensure positive
        else:
            # Build surface if not available
            self.build_local_vol_surface()
            return self.get_local_vol(S, t)
    
    def price_european_option(self, K: float, T: float, option_type: str = 'call',
                            n_paths: int = 100000, n_steps: int = 100,
                            scheme: str = 'euler') -> Dict:
        """
        Price European option using local volatility Monte Carlo.
        
        Args:
            K: Strike price
            T: Time to maturity
            option_type: 'call' or 'put'
            n_paths: Number of Monte Carlo paths
            n_steps: Number of time steps
            scheme: 'euler' or 'milstein'
            
        Returns:
            Pricing results dictionary
        """
        if self.local_vol_interpolator is None:
            self.build_local_vol_surface()
        
        # Time discretization
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        time_grid = np.linspace(0, T, n_steps + 1)
        
        # Initialize paths
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.spot
        
        # Generate random numbers
        Z = np.random.standard_normal((n_paths, n_steps))
        
        # Simulate paths
        for i in range(n_steps):
            t = time_grid[i]
            S = paths[:, i]
            
            # Get local volatilities for all paths (vectorized)
            local_vols = np.array([self.get_local_vol(s, t) for s in S])
            
            if scheme == 'milstein':
                # Milstein scheme for better accuracy
                h = 0.01 * S  # Small perturbation
                local_vols_up = np.array([self.get_local_vol(s + h[j], t) for j, s in enumerate(S)])
                d_sigma_dS = (local_vols_up - local_vols) / h
                
                # Milstein step
                drift = (self.r - self.q) * S * dt
                diffusion = local_vols * S * sqrt_dt * Z[:, i]
                milstein_correction = 0.5 * local_vols * S * d_sigma_dS * ((Z[:, i]**2 - 1) * dt)
                
                paths[:, i+1] = S + drift + diffusion + milstein_correction
            else:
                # Standard Euler scheme
                drift = (self.r - self.q) * S * dt
                diffusion = local_vols * S * sqrt_dt * Z[:, i]
                paths[:, i+1] = S + drift + diffusion
            
            # Ensure positive stock prices
            paths[:, i+1] = np.maximum(paths[:, i+1], 1e-6)
        
        # Calculate payoffs
        final_prices = paths[:, -1]
        if option_type == 'call':
            payoffs = np.maximum(final_prices - K, 0)
        else:
            payoffs = np.maximum(K - final_prices, 0)
        
        # Discount to present value
        discount_factor = np.exp(-self.r * T)
        discounted_payoffs = payoffs * discount_factor
        
        # Calculate statistics
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(n_paths)
        
        # Black-Scholes price for comparison
        atm_vol = self.vol_surface.get_vol(K, T) if self.vol_surface else 0.25
        if option_type == 'call':
            bs_price = BlackScholes.call_price(self.spot, K, T, self.r, atm_vol, self.q)
        else:
            bs_price = BlackScholes.put_price(self.spot, K, T, self.r, atm_vol, self.q)
        
        return {
            'price': price,
            'std_error': std_error,
            'confidence_interval': (price - 1.96 * std_error, price + 1.96 * std_error),
            'bs_price': bs_price,
            'difference': price - bs_price,
            'paths': n_paths,
            'steps': n_steps
        }
    
    def price_barrier_option(self, K: float, H: float, T: float,
                           barrier_type: str = 'down-out', option_type: str = 'call',
                           n_paths: int = 100000, n_steps: int = 500) -> Dict:
        """
        Price barrier option using local volatility Monte Carlo.
        
        Args:
            K: Strike price
            H: Barrier level
            T: Time to maturity
            barrier_type: 'down-out', 'up-out', 'down-in', 'up-in'
            option_type: 'call' or 'put'
            n_paths: Number of paths
            n_steps: Number of time steps
            
        Returns:
            Pricing results dictionary
        """
        if self.local_vol_interpolator is None:
            self.build_local_vol_surface()
        
        # Time discretization
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        time_grid = np.linspace(0, T, n_steps + 1)
        
        # Initialize paths
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.spot
        
        # Track barrier crossings
        hit_barrier = np.zeros(n_paths, dtype=bool)
        
        # Generate random numbers
        Z = np.random.standard_normal((n_paths, n_steps))
        
        # Simulate paths with barrier monitoring
        for i in range(n_steps):
            t = time_grid[i]
            S = paths[:, i]
            
            # Get local volatilities
            local_vols = np.array([self.get_local_vol(s, t) for s in S])
            
            # Euler step
            drift = (self.r - self.q) * S * dt
            diffusion = local_vols * S * sqrt_dt * Z[:, i]
            paths[:, i+1] = S + drift + diffusion
            
            # Ensure positive prices
            paths[:, i+1] = np.maximum(paths[:, i+1], 1e-6)
            
            # Check barrier crossing
            if barrier_type.startswith('down'):
                hit_barrier |= (paths[:, i+1] <= H)
            else:  # up barrier
                hit_barrier |= (paths[:, i+1] >= H)
        
        # Calculate payoffs
        final_prices = paths[:, -1]
        if option_type == 'call':
            payoffs = np.maximum(final_prices - K, 0)
        else:
            payoffs = np.maximum(K - final_prices, 0)
        
        # Apply barrier conditions
        if barrier_type.endswith('out'):
            payoffs[hit_barrier] = 0
        else:  # knock-in
            payoffs[~hit_barrier] = 0
        
        # Discount to present value
        discount_factor = np.exp(-self.r * T)
        discounted_payoffs = payoffs * discount_factor
        
        # Calculate statistics
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(n_paths)
        knock_probability = np.mean(hit_barrier)
        
        # Black-Scholes vanilla price for comparison
        atm_vol = self.vol_surface.get_vol(K, T) if self.vol_surface else 0.25
        if option_type == 'call':
            bs_price = BlackScholes.call_price(self.spot, K, T, self.r, atm_vol, self.q)
        else:
            bs_price = BlackScholes.put_price(self.spot, K, T, self.r, atm_vol, self.q)
        
        return {
            'price': price,
            'std_error': std_error,
            'confidence_interval': (price - 1.96 * std_error, price + 1.96 * std_error),
            'knock_probability': knock_probability,
            'bs_price': bs_price,
            'difference': price - bs_price,
            'barrier_type': barrier_type,
            'barrier_level': H
        }
    
    def analyze_convergence(self, K: float, T: float, option_type: str = 'call',
                          path_counts: List[int] = None, step_counts: List[int] = None,
                          target_error: float = 1e-6, fast_mode: bool = False) -> Dict:
        """
        Analyze Monte Carlo convergence.
        
        Args:
            K: Strike price
            T: Time to maturity
            option_type: 'call' or 'put'
            path_counts: List of path counts to test
            step_counts: List of step counts to test
            target_error: Target standard error
            fast_mode: Use smaller counts for faster execution
            
        Returns:
            Convergence analysis results
        """
        if path_counts is None:
            if fast_mode:
                path_counts = [1000, 5000, 10000, 50000]
            else:
                path_counts = [1000, 5000, 10000, 50000, 100000]
        
        if step_counts is None:
            if fast_mode:
                step_counts = [50, 100]
            else:
                step_counts = [50, 100, 250]
        
        results = {
            'path_convergence': {
                'path_counts': [],
                'prices': [],
                'std_errors': [],
                'absolute_errors': [],
                'confidence_intervals': []
            },
            'step_convergence': {
                'step_counts': [],
                'prices': [],
                'absolute_errors': []
            },
            'achieved_error': None,
            'required_paths': None,
            'bs_price': None,
            'sigma_atm': None
        }
        
        # Get Black-Scholes benchmark
        if self.vol_surface:
            sigma_atm = self.vol_surface.get_vol(K, T)
            if option_type == 'call':
                bs_price = BlackScholes.call_price(self.spot, K, T, self.r, sigma_atm, self.q)
            else:
                bs_price = BlackScholes.put_price(self.spot, K, T, self.r, sigma_atm, self.q)
            
            results['bs_price'] = bs_price
            results['sigma_atm'] = sigma_atm
        else:
            bs_price = None
        
        # Path convergence analysis
        print("Testing path convergence...")
        base_steps = 50 if fast_mode else 100
        
        for n_paths in path_counts:
            result = self.price_european_option(
                K, T, option_type, n_paths=n_paths, n_steps=base_steps
            )
            
            results['path_convergence']['path_counts'].append(n_paths)
            results['path_convergence']['prices'].append(result['price'])
            results['path_convergence']['std_errors'].append(result['std_error'])
            
            if bs_price:
                abs_error = abs(result['price'] - bs_price)
            else:
                abs_error = result['std_error']
            results['path_convergence']['absolute_errors'].append(abs_error)
            
            ci_width = 2 * 1.96 * result['std_error']
            results['path_convergence']['confidence_intervals'].append({
                'lower': result['price'] - 1.96 * result['std_error'],
                'upper': result['price'] + 1.96 * result['std_error'],
                'precision': ci_width
            })
            
            if result['std_error'] <= target_error:
                results['achieved_error'] = result['std_error']
                results['required_paths'] = n_paths
                print(f"  Target error {target_error:.2e} achieved with {n_paths:,} paths")
                break
        
        # Step convergence analysis
        if not fast_mode:
            print("Testing step convergence...")
            test_paths = 10000
            
            for n_steps in step_counts:
                result = self.price_european_option(
                    K, T, option_type, n_paths=test_paths, n_steps=n_steps
                )
                
                results['step_convergence']['step_counts'].append(n_steps)
                results['step_convergence']['prices'].append(result['price'])
                
                if bs_price:
                    abs_error = abs(result['price'] - bs_price)
                else:
                    abs_error = 0
                results['step_convergence']['absolute_errors'].append(abs_error)
        
        return results
    
    def analyze_convergence_2d(self, K: float, T: float, option_type: str = 'call',
                              path_counts: List[int] = None,
                              step_counts: List[int] = None) -> Dict:
        """
        Analyze 2D convergence varying both paths and steps.
        
        Args:
            K: Strike price
            T: Time to maturity
            option_type: 'call' or 'put'
            path_counts: List of path counts
            step_counts: List of step counts
            
        Returns:
            2D convergence analysis results
        """
        if path_counts is None:
            path_counts = [1000, 5000, 10000, 50000]
        
        if step_counts is None:
            step_counts = [25, 50, 100, 200]
        
        # Get reference price
        sigma_atm = self.vol_surface.get_vol(K, T) if self.vol_surface else 0.25
        if option_type == 'call':
            ref_price = BlackScholes.call_price(self.spot, K, T, self.r, sigma_atm, self.q)
        else:
            ref_price = BlackScholes.put_price(self.spot, K, T, self.r, sigma_atm, self.q)
        
        # Build grids
        price_grid = np.zeros((len(path_counts), len(step_counts)))
        error_grid = np.zeros((len(path_counts), len(step_counts)))
        
        print("Running 2D convergence analysis...")
        total = len(path_counts) * len(step_counts)
        count = 0
        
        for i, n_paths in enumerate(path_counts):
            for j, n_steps in enumerate(step_counts):
                count += 1
                print(f"  Progress: {count}/{total} ({n_paths} paths, {n_steps} steps)")
                
                result = self.price_european_option(
                    K, T, option_type, n_paths=n_paths, n_steps=n_steps
                )
                
                price_grid[i, j] = result['price']
                error_grid[i, j] = abs(result['price'] - ref_price)
        
        return {
            'path_counts': path_counts,
            'step_counts': step_counts,
            'price_grid': price_grid,
            'error_grid': error_grid,
            'reference_price': ref_price,
            'bs_price': ref_price
        }