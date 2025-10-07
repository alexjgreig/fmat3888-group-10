import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Union
import warnings
warnings.filterwarnings('ignore')


def calculate_mid_price(bid: float, ask: float, last: float) -> float:
    """
    Calculate mid price from bid/ask spread or use last price if bid/ask unavailable.
    
    Args:
        bid: Bid price
        ask: Ask price  
        last: Last traded price
        
    Returns:
        Mid price (average of bid/ask) or last price if bid/ask not available
    """
    if pd.notna(bid) and pd.notna(ask) and bid > 0 and ask > 0:
        return (bid + ask) / 2
    elif pd.notna(last) and last > 0:
        return last
    else:
        return None


def get_option_expirations(ticker: str) -> List[str]:
    """
    Get all available option expiration dates for a given ticker.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        List of expiration dates as strings (YYYY-MM-DD format)
    """
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        return expirations
    except Exception as e:
        print(f"Error fetching expirations for {ticker}: {e}")
        return []


def get_option_chain(ticker: str, expiration: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Get option chain data for a specific ticker and expiration date.
    
    Args:
        ticker: Stock ticker symbol
        expiration: Expiration date (YYYY-MM-DD format). If None, uses nearest expiration.
        
    Returns:
        Dictionary with 'calls' and 'puts' DataFrames containing option data with mid prices
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Get current stock price
        current_price = stock.info.get('currentPrice') or stock.info.get('regularMarketPrice')
        
        # If no expiration specified, use the nearest one
        if expiration is None:
            expirations = stock.options
            if not expirations:
                print(f"No options available for {ticker}")
                return {'calls': pd.DataFrame(), 'puts': pd.DataFrame()}
            expiration = expirations[0]
        
        # Get option chain
        opt_chain = stock.option_chain(expiration)
        
        # Process calls
        calls_df = opt_chain.calls.copy()
        if not calls_df.empty:
            calls_df['mid'] = calls_df.apply(
                lambda row: calculate_mid_price(row['bid'], row['ask'], row['lastPrice']), 
                axis=1
            )
            calls_df['optionType'] = 'CALL'
            calls_df['underlyingPrice'] = current_price
            calls_df['expiration'] = expiration
            calls_df['ticker'] = ticker
            
        # Process puts  
        puts_df = opt_chain.puts.copy()
        if not puts_df.empty:
            puts_df['mid'] = puts_df.apply(
                lambda row: calculate_mid_price(row['bid'], row['ask'], row['lastPrice']), 
                axis=1
            )
            puts_df['optionType'] = 'PUT'
            puts_df['underlyingPrice'] = current_price
            puts_df['expiration'] = expiration
            puts_df['ticker'] = ticker
        
        return {
            'calls': calls_df,
            'puts': puts_df,
            'underlying_price': current_price,
            'expiration': expiration
        }
        
    except Exception as e:
        print(f"Error fetching option chain for {ticker}: {e}")
        return {'calls': pd.DataFrame(), 'puts': pd.DataFrame()}


def pull_options_data(
    tickers: Union[str, List[str]], 
    specific_expiry: Optional[str] = None,
    expirations_to_fetch: int = 1
) -> pd.DataFrame:
    """
    Pull options data for multiple tickers.
    
    Args:
        tickers: Single ticker or list of ticker symbols
        specific_expiry: Specific expiration date to fetch (YYYY-MM-DD)
        expirations_to_fetch: Number of expiration dates to fetch per ticker (if specific_expiry not provided)
        
    Returns:
        Combined DataFrame with all option data
    """
    # Convert single ticker to list
    if isinstance(tickers, str):
        tickers = [tickers]
    
    all_options = []
    
    for ticker in tickers:
        print(f"\nFetching options for {ticker}...")
        
        if specific_expiry:
            # Fetch specific expiration
            chain_data = get_option_chain(ticker, specific_expiry)
            if not chain_data['calls'].empty:
                all_options.append(chain_data['calls'])
            if not chain_data['puts'].empty:
                all_options.append(chain_data['puts'])
        else:
            # Fetch multiple expirations
            expirations = get_option_expirations(ticker)
            
            for i, exp in enumerate(expirations[:expirations_to_fetch]):
                print(f"  Fetching expiration: {exp}")
                chain_data = get_option_chain(ticker, exp)
                
                if not chain_data['calls'].empty:
                    all_options.append(chain_data['calls'])
                if not chain_data['puts'].empty:
                    all_options.append(chain_data['puts'])
    
    # Combine all data
    if all_options:
        combined_df = pd.concat(all_options, ignore_index=True)
        
        # Select and reorder important columns
        important_cols = [
            'ticker', 'optionType', 'expiration', 'strike', 
            'bid', 'ask', 'mid', 'lastPrice', 
            'volume', 'openInterest', 'impliedVolatility',
            'underlyingPrice', 'inTheMoney'
        ]
        
        # Keep only columns that exist
        cols_to_keep = [col for col in important_cols if col in combined_df.columns]
        combined_df = combined_df[cols_to_keep]
        
        return combined_df
    else:
        return pd.DataFrame()


def get_atm_options(ticker: str, expiration: Optional[str] = None) -> pd.DataFrame:
    """
    Get at-the-money (ATM) options for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        expiration: Expiration date (YYYY-MM-DD format)
        
    Returns:
        DataFrame with ATM calls and puts
    """
    chain_data = get_option_chain(ticker, expiration)
    
    if chain_data['calls'].empty and chain_data['puts'].empty:
        return pd.DataFrame()
    
    underlying_price = chain_data['underlying_price']
    
    # Find ATM strikes (closest to underlying price)
    atm_options = []
    
    for df, opt_type in [(chain_data['calls'], 'CALL'), (chain_data['puts'], 'PUT')]:
        if not df.empty:
            df['distance'] = abs(df['strike'] - underlying_price)
            atm_strike_idx = df['distance'].idxmin()
            atm_option = df.loc[atm_strike_idx:atm_strike_idx]
            atm_options.append(atm_option)
    
    if atm_options:
        return pd.concat(atm_options, ignore_index=True)
    else:
        return pd.DataFrame()


def main():
    """
    Example usage of the option data pulling functions.
    """
    print("=" * 60)
    print("Yahoo Finance Options Data Pull")
    print("=" * 60)
    
    # Example 1: Get option expirations for a ticker
    ticker = "AAPL"
    print(f"\n1. Available expirations for {ticker}:")
    expirations = get_option_expirations(ticker)
    print(f"   Found {len(expirations)} expiration dates")
    if expirations:
        print(f"   Next 5 expirations: {expirations[:5]}")
    
    # Example 2: Get option chain for nearest expiration
    print(f"\n2. Option chain for {ticker} (nearest expiration):")
    chain_data = get_option_chain(ticker)
    if chain_data['calls'].shape[0] > 0:
        print(f"   Calls: {chain_data['calls'].shape[0]} contracts")
        print(f"   Puts: {chain_data['puts'].shape[0]} contracts")
        print(f"   Underlying Price: ${chain_data['underlying_price']:.2f}")
        
        # Show sample of calls with mid prices
        print("\n   Sample CALL options:")
        sample_calls = chain_data['calls'][['strike', 'bid', 'ask', 'mid', 'volume', 'openInterest']].head(5)
        print(sample_calls.to_string(index=False))
    
    # Example 3: Get ATM options
    print(f"\n3. At-the-money options for {ticker}:")
    atm_options = get_atm_options(ticker)
    if not atm_options.empty:
        print(atm_options[['optionType', 'strike', 'bid', 'ask', 'mid', 'impliedVolatility']].to_string(index=False))
    
    # Example 4: Pull data for multiple tickers
    print("\n4. Fetching options for multiple tickers:")
    tickers = ["AAPL", "MSFT", "GOOGL"]
    combined_data = pull_options_data(tickers, expirations_to_fetch=1)
    
    if not combined_data.empty:
        print(f"\n   Total options fetched: {len(combined_data)}")
        print("\n   Summary by ticker and type:")
        summary = combined_data.groupby(['ticker', 'optionType']).agg({
            'strike': 'count',
            'volume': 'sum',
            'openInterest': 'sum'
        }).rename(columns={'strike': 'contracts'})
        print(summary)
        
        # Show sample of data with mid prices
        print("\n   Sample of combined data:")
        sample_cols = ['ticker', 'optionType', 'expiration', 'strike', 'bid', 'ask', 'mid', 'volume']
        print(combined_data[sample_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()