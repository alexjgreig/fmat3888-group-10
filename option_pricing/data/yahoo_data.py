"""
Enhanced Yahoo Finance data pulling module for PLTR options.
Includes caching, validation, and robust error handling.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Union, Tuple
import warnings
import pickle
from pathlib import Path
import hashlib
from diskcache import Cache
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.black_scholes import BlackScholes

warnings.filterwarnings('ignore')


class OptionDataPuller:
    """Enhanced option data puller with caching and validation."""
    
    def __init__(self, ticker: str = "PLTR", cache_dir: str = ".cache", 
                 risk_free_rate: float = 0.05):
        """
        Initialize the option data puller.
        
        Args:
            ticker: Stock ticker symbol (default: PLTR for Palantir)
            cache_dir: Directory for caching data
            risk_free_rate: Risk-free rate for IV calculation
        """
        self.ticker = ticker
        self.cache = Cache(cache_dir)
        self.cache_expiry = 3600  # 1 hour cache expiry
        self.risk_free_rate = risk_free_rate
        self._validate_ticker()
        
    def _validate_ticker(self):
        """Validate that the ticker exists and has options."""
        try:
            stock = yf.Ticker(self.ticker)
            if not stock.options:
                raise ValueError(f"No options available for {self.ticker}")
        except Exception as e:
            raise ValueError(f"Invalid ticker {self.ticker}: {e}")
    
    def _get_cache_key(self, method: str, *args, **kwargs) -> str:
        """Generate a cache key based on method and arguments."""
        key_str = f"{self.ticker}_{method}_{args}_{kwargs}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def calculate_implied_volatility(self, option_price: float, spot: float, strike: float,
                                    time_to_expiry: float, option_type: str,
                                    dividend_yield: float = 0.0) -> Optional[float]:
        """
        Calculate implied volatility from option price using Black-Scholes inverse.
        
        Args:
            option_price: Market price of the option
            spot: Current stock price
            strike: Strike price
            time_to_expiry: Time to expiration in years
            option_type: 'call' or 'put'
            dividend_yield: Dividend yield (default 0)
            
        Returns:
            Implied volatility or None if cannot be calculated
        """
        if option_price <= 0 or time_to_expiry <= 0:
            return None
            
        try:
            iv = BlackScholes.implied_volatility(
                price=option_price,
                S=spot,
                K=strike,
                T=time_to_expiry,
                r=self.risk_free_rate,
                q=dividend_yield,
                option_type=option_type,
                max_iterations=100,
                tolerance=1e-6
            )
            
            # Validate IV is reasonable
            if iv is not None and 0.01 < iv < 5.0:
                return iv
            return None
            
        except Exception as e:
            print(f"Error calculating IV: {e}")
            return None
    
    def calculate_mid_price(self, bid: float, ask: float, last: float) -> Optional[float]:
        """
        Calculate mid price from bid/ask spread with validation.
        
        Args:
            bid: Bid price
            ask: Ask price  
            last: Last traded price
            
        Returns:
            Mid price or None if invalid
        """
        # Check for valid bid/ask spread
        if pd.notna(bid) and pd.notna(ask) and bid > 0 and ask > 0:
            # Validate spread is reasonable (not too wide)
            spread = (ask - bid) / ask
            if spread < 0.5:  # Reject if spread > 50%
                return (bid + ask) / 2
        
        # Fall back to last price if available
        if pd.notna(last) and last > 0:
            return last
        
        return None
    
    def get_current_price(self) -> float:
        """Get current stock price with caching."""
        cache_key = self._get_cache_key("current_price")
        
        # Check cache
        cached_price = self.cache.get(cache_key)
        if cached_price is not None:
            return cached_price
        
        try:
            stock = yf.Ticker(self.ticker)
            price = stock.info.get('currentPrice') or stock.info.get('regularMarketPrice')
            
            if price is None:
                # Try getting from history
                hist = stock.history(period="1d")
                if not hist.empty:
                    price = hist['Close'].iloc[-1]
            
            if price is not None:
                self.cache.set(cache_key, price, expire=self.cache_expiry)
                return price
            else:
                raise ValueError("Could not retrieve current price")
                
        except Exception as e:
            raise RuntimeError(f"Error fetching current price: {e}")
    
    def get_option_expirations(self) -> List[str]:
        """Get all available option expiration dates."""
        cache_key = self._get_cache_key("expirations")
        
        # Check cache
        cached_expirations = self.cache.get(cache_key)
        if cached_expirations is not None:
            return cached_expirations
        
        try:
            stock = yf.Ticker(self.ticker)
            expirations = list(stock.options)
            
            if expirations:
                self.cache.set(cache_key, expirations, expire=self.cache_expiry * 24)  # Cache for 24 hours
            
            return expirations
        except Exception as e:
            print(f"Error fetching expirations: {e}")
            return []
    
    def get_option_chain(self, expiration: Optional[str] = None) -> Dict:
        """
        Get option chain data for a specific expiration.
        
        Args:
            expiration: Expiration date (YYYY-MM-DD format). If None, uses nearest expiration.
            
        Returns:
            Dictionary with calls, puts, underlying price, and metadata
        """
        # Use nearest expiration if not specified
        if expiration is None:
            expirations = self.get_option_expirations()
            if not expirations:
                return {'calls': pd.DataFrame(), 'puts': pd.DataFrame()}
            expiration = expirations[0]
        
        cache_key = self._get_cache_key("option_chain", expiration)
        
        # Check cache
        cached_chain = self.cache.get(cache_key)
        if cached_chain is not None:
            return cached_chain
        
        try:
            stock = yf.Ticker(self.ticker)
            current_price = self.get_current_price()
            
            # Get option chain
            opt_chain = stock.option_chain(expiration)
            
            # Process calls
            calls_df = self._process_option_data(
                opt_chain.calls, 'CALL', current_price, expiration
            )
            
            # Process puts  
            puts_df = self._process_option_data(
                opt_chain.puts, 'PUT', current_price, expiration
            )
            
            result = {
                'calls': calls_df,
                'puts': puts_df,
                'underlying_price': current_price,
                'expiration': expiration,
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache the result
            self.cache.set(cache_key, result, expire=self.cache_expiry)
            
            return result
            
        except Exception as e:
            print(f"Error fetching option chain: {e}")
            return {'calls': pd.DataFrame(), 'puts': pd.DataFrame()}
    
    def _process_option_data(self, df: pd.DataFrame, option_type: str, 
                            underlying_price: float, expiration: str) -> pd.DataFrame:
        """Process and enhance option data."""
        if df.empty:
            return df
        
        df = df.copy()
        
        # Calculate mid prices
        df['mid'] = df.apply(
            lambda row: self.calculate_mid_price(row['bid'], row['ask'], row['lastPrice']), 
            axis=1
        )
        
        # Add metadata
        df['optionType'] = option_type
        df['underlyingPrice'] = underlying_price
        df['expiration'] = expiration
        df['ticker'] = self.ticker
        
        # Calculate moneyness
        df['moneyness'] = df['strike'] / underlying_price
        
        # Calculate time to expiration in years
        exp_date = pd.to_datetime(expiration)
        today = pd.Timestamp.now().normalize()
        df['timeToExpiry'] = (exp_date - today).days / 365.25
        
        # Filter out options with invalid data
        df = df[df['mid'].notna()]
        
        # Calculate IMPLIED VOLATILITY from option prices (not using Yahoo's IV)
        print(f"    Calculating implied volatilities for {len(df)} {option_type.lower()}s...")
        df['calculatedIV'] = df.apply(
            lambda row: self.calculate_implied_volatility(
                option_price=row['mid'],
                spot=underlying_price,
                strike=row['strike'],
                time_to_expiry=row['timeToExpiry'],
                option_type=option_type.lower(),
                dividend_yield=0.0
            ),
            axis=1
        )
        
        # Keep Yahoo's IV for comparison but rename it
        df['yahooIV'] = df['impliedVolatility']
        
        # Use our calculated IV as the main impliedVolatility
        df['impliedVolatility'] = df['calculatedIV']
        
        # Add liquidity score (combination of volume and open interest)
        df['liquidityScore'] = df['volume'].fillna(0) + df['openInterest'].fillna(0) * 0.1
        
        return df
    
    def get_liquid_options(self, min_volume: int = 10, min_oi: int = 50,
                           num_expirations: int = 5) -> pd.DataFrame:
        """
        Get liquid options filtered by volume and open interest.
        
        Args:
            min_volume: Minimum daily volume
            min_oi: Minimum open interest
            num_expirations: Number of expirations to fetch
            
        Returns:
            DataFrame with liquid options
        """
        expirations = self.get_option_expirations()[:num_expirations]
        all_options = []
        
        for exp in expirations:
            chain_data = self.get_option_chain(exp)
            
            for df in [chain_data['calls'], chain_data['puts']]:
                if not df.empty:
                    # Filter by liquidity
                    liquid_df = df[
                        (df['volume'].fillna(0) >= min_volume) | 
                        (df['openInterest'].fillna(0) >= min_oi)
                    ]
                    all_options.append(liquid_df)
        
        if all_options:
            return pd.concat(all_options, ignore_index=True)
        return pd.DataFrame()
    
    def get_atm_options(self, expiration: Optional[str] = None, 
                       strike_range: float = 0.1) -> pd.DataFrame:
        """
        Get at-the-money options within a specified strike range.
        
        Args:
            expiration: Expiration date
            strike_range: Percentage range around ATM (e.g., 0.1 = ±10%)
            
        Returns:
            DataFrame with ATM options
        """
        chain_data = self.get_option_chain(expiration)
        underlying_price = chain_data['underlying_price']
        
        atm_options = []
        
        for df in [chain_data['calls'], chain_data['puts']]:
            if not df.empty:
                # Filter options within strike range
                lower_bound = underlying_price * (1 - strike_range)
                upper_bound = underlying_price * (1 + strike_range)
                
                atm_df = df[
                    (df['strike'] >= lower_bound) & 
                    (df['strike'] <= upper_bound)
                ]
                atm_options.append(atm_df)
        
        if atm_options:
            return pd.concat(atm_options, ignore_index=True)
        return pd.DataFrame()
    
    def get_full_surface_data(self, num_expirations: int = 10,
                             moneyness_range: Tuple[float, float] = (0.7, 1.3)) -> pd.DataFrame:
        """
        Get comprehensive option data for volatility surface construction.
        
        Args:
            num_expirations: Number of expiration dates to fetch
            moneyness_range: Range of moneyness to include
            
        Returns:
            DataFrame with filtered option data suitable for vol surface
        """
        expirations = self.get_option_expirations()[:num_expirations]
        all_options = []
        
        print(f"Fetching {len(expirations)} expirations for {self.ticker}...")
        
        for i, exp in enumerate(expirations):
            print(f"  Processing expiration {i+1}/{len(expirations)}: {exp}")
            chain_data = self.get_option_chain(exp)
            
            for df in [chain_data['calls'], chain_data['puts']]:
                if not df.empty:
                    # Filter by moneyness range
                    filtered_df = df[
                        (df['moneyness'] >= moneyness_range[0]) & 
                        (df['moneyness'] <= moneyness_range[1])
                    ]
                    
                    # Only keep options with valid calculated implied volatility
                    filtered_df = filtered_df[
                        (filtered_df['calculatedIV'].notna()) & 
                        (filtered_df['calculatedIV'] > 0.01) &
                        (filtered_df['calculatedIV'] < 5.0)  # Remove extreme IVs
                    ]
                    
                    # Update impliedVolatility to use our calculated values
                    filtered_df['impliedVolatility'] = filtered_df['calculatedIV']
                    
                    all_options.append(filtered_df)
        
        if all_options:
            full_data = pd.concat(all_options, ignore_index=True)
            
            # Sort by expiration and strike
            full_data = full_data.sort_values(['expiration', 'strike'])
            
            # Add some statistics
            print(f"\nData Summary:")
            print(f"  Total options: {len(full_data)}")
            print(f"  Unique expirations: {full_data['expiration'].nunique()}")
            print(f"  Strike range: ${full_data['strike'].min():.2f} - ${full_data['strike'].max():.2f}")
            print(f"  IV range: {full_data['impliedVolatility'].min():.2%} - {full_data['impliedVolatility'].max():.2%}")
            
            return full_data
        
        return pd.DataFrame()
    
    def save_data(self, data: pd.DataFrame, filename: str = None):
        """Save option data to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"option_data_{self.ticker}_{timestamp}.pkl"
        
        data.to_pickle(filename)
        print(f"Data saved to {filename}")
    
    def load_data(self, filename: str) -> pd.DataFrame:
        """Load option data from file."""
        return pd.read_pickle(filename)


def main():
    """Example usage of the enhanced option data puller."""
    print("=" * 60)
    print("Enhanced PLTR Options Data Pull")
    print("=" * 60)
    
    # Initialize data puller for PLTR
    puller = OptionDataPuller("PLTR")
    
    # Get current price
    print(f"\nCurrent PLTR Price: ${puller.get_current_price():.2f}")
    
    # Get available expirations
    expirations = puller.get_option_expirations()
    print(f"\nAvailable expirations: {len(expirations)}")
    if expirations:
        print(f"Next 5 expirations: {expirations[:5]}")
    
    # Get liquid options
    print("\nFetching liquid options...")
    liquid_options = puller.get_liquid_options(min_volume=50, min_oi=100, num_expirations=3)
    
    if not liquid_options.empty:
        print(f"Found {len(liquid_options)} liquid options")
        
        # Show summary
        print("\nLiquid Options Summary:")
        summary = liquid_options.groupby(['optionType', 'expiration']).agg({
            'strike': 'count',
            'volume': 'sum',
            'openInterest': 'sum',
            'impliedVolatility': 'mean'
        }).round(4)
        print(summary)
    
    # Get full surface data
    print("\nFetching full surface data...")
    surface_data = puller.get_full_surface_data(num_expirations=5)
    
    if not surface_data.empty:
        # Save the data
        puller.save_data(surface_data)
        
        # Show sample
        print("\nSample of surface data:")
        sample_cols = ['ticker', 'optionType', 'expiration', 'strike', 'mid', 
                      'impliedVolatility', 'moneyness', 'liquidityScore']
        print(surface_data[sample_cols].head(10))


if __name__ == "__main__":
    main()