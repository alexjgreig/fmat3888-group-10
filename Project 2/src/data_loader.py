"""
Data Loader Module for FMAT3888 Project 2
Handles loading and preprocessing of historical asset returns data
"""

import os
import pandas as pd
import numpy as np
import warnings
from typing import Tuple, Dict, List
from datetime import datetime

warnings.filterwarnings('ignore')


class AssetDataLoader:
    """Load and preprocess historical asset returns data from Excel file"""

    def __init__(self, file_path: str):
        """
        Initialize data loader

        Args:
            file_path: Path to the Excel file containing historical data
        """
        self.file_path = file_path
        self.asset_classes = [
            'Australian Listed Equity [G]',
            "Int'l Listed Equity (Hedged) [G]",
            "Int'l Listed Equity (Unhedged) [G]",
            'Australian Listed Property [G]',
            "Int'l Listed Property [G]",
            "Int'l Listed Infrastructure [G]",
            'Australian Fixed Income [D]',
            "Int'l Fixed Income (Hedged) [D]",
            'Cash [D]'
        ]
        self.growth_assets = self.asset_classes[:6]  # First 6 are growth
        self.defensive_assets = self.asset_classes[6:]  # Last 3 are defensive
        # Bloomberg tickers that correspond to each asset class when using the Market Data sheet
        self.asset_ticker_aliases = {
            'Australian Listed Equity [G]': ['ASA52'],
            "Int'l Listed Equity (Hedged) [G]": ['NDDLWI'],
            "Int'l Listed Equity (Unhedged) [G]": ['NDLEEGF'],
            'Australian Listed Property [G]': ['RDAU'],
            "Int'l Listed Property [G]": ['FDCIISAH'],
            "Int'l Listed Infrastructure [G]": ['HEDGNAV'],
            'Australian Fixed Income [D]': ['BACM0'],
            "Int'l Fixed Income (Hedged) [D]": ['LGTRTRAH', 'H03432AU'],
            'Cash [D]': ['BAUBIL']
        }

    def load_data(self) -> pd.DataFrame:
        """
        Load and clean the historical returns data

        Returns:
            DataFrame with monthly returns for each asset class
        """
        try:
            excel_file = pd.ExcelFile(self.file_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Data file not found: {self.file_path}") from exc

        if 'Market Data' in excel_file.sheet_names:
            raw_df = excel_file.parse('Market Data')
            returns_df = self._process_market_data_sheet(raw_df)
        elif 'Historical Returns' in excel_file.sheet_names:
            raw_df = excel_file.parse('Historical Returns', header=None)
            returns_df = self._process_historical_returns_sheet(raw_df)
        else:
            available_sheets = ', '.join(excel_file.sheet_names)
            raise ValueError(
                "Expected to find either a 'Market Data' or 'Historical Returns' sheet "
                f"in {self.file_path}. Available sheets: {available_sheets}"
            )

        # Store the processed data
        self.returns_data = returns_df

        return returns_df

    def _process_market_data_sheet(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the Market Data sheet into the standard returns DataFrame.
        """
        # Normalise column labels for easier matching
        df = df.copy()
        df.columns = [
            col.strip() if isinstance(col, str) else col
            for col in df.columns
        ]

        if df.empty:
            raise ValueError("Market Data sheet is empty.")

        # Extract and clean dates from the first column
        date_series = df.iloc[:, 0]
        dates = date_series.apply(self._convert_excel_date)
        df.index = dates

        # Build the returns DataFrame using ticker aliases
        clean_data = {}
        for asset in self.asset_classes:
            ticker_aliases = self.asset_ticker_aliases.get(asset, [])
            col_name = self._match_market_column(df.columns, ticker_aliases)
            if col_name is None:
                aliases = ', '.join(ticker_aliases) or 'N/A'
                raise KeyError(
                    f"Could not locate a column for asset '{asset}' "
                    f"using ticker aliases: {aliases}"
                )
            clean_data[asset] = pd.to_numeric(df[col_name], errors='coerce')

        returns_df = pd.DataFrame(clean_data, index=dates)

        # Drop rows without a valid date
        returns_df = returns_df[returns_df.index.notna()]

        # Sort by date and clean duplicates
        returns_df = returns_df.sort_index()
        returns_df = returns_df[~returns_df.index.duplicated(keep='first')]

        # Drop rows where all asset returns are NaN
        returns_df = returns_df.dropna(how='all')

        # Fill remaining gaps
        returns_df = returns_df.ffill().bfill()

        return returns_df

    def _process_historical_returns_sheet(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Legacy support for the Historical Returns sheet structure.
        """
        header_row = 3
        data_start_row = 4

        dates_col = df.iloc[data_start_row:, 1]

        dates = []
        for date_val in dates_col:
            if pd.notna(date_val):
                try:
                    if isinstance(date_val, (int, float)):
                        date = pd.Timestamp('1899-12-30') + pd.Timedelta(days=float(date_val))
                    else:
                        date = pd.to_datetime(date_val)
                    dates.append(date)
                except Exception:
                    dates.append(None)
            else:
                dates.append(None)

        col_mapping = {
            2: 'Australian Listed Equity [G]',
            3: "Int'l Listed Equity (Hedged) [G]",
            4: "Int'l Listed Equity (Unhedged) [G]",
            5: 'Australian Listed Property [G]',
            6: "Int'l Listed Property [G]",
            7: "Int'l Listed Infrastructure [G]",
            8: 'Australian Fixed Income [D]',
            9: "Int'l Fixed Income (Hedged) [D]",
            10: 'Cash [D]'
        }

        clean_data = {}
        for col_idx, asset_name in col_mapping.items():
            values = df.iloc[data_start_row:, col_idx].values
            numeric_values = pd.to_numeric(values, errors='coerce')
            clean_data[asset_name] = numeric_values[:len(dates)]

        returns_df = pd.DataFrame(clean_data, index=dates)

        returns_df = returns_df[returns_df.index.notna()]
        returns_df = returns_df.sort_index()
        returns_df = returns_df[~returns_df.index.duplicated(keep='first')]
        returns_df = returns_df.ffill().bfill()

        return returns_df

    @staticmethod
    def _convert_excel_date(value):
        """Convert Excel serial dates or strings into pandas timestamps."""
        if pd.isna(value):
            return pd.NaT
        if isinstance(value, (int, float)):
            try:
                return pd.Timestamp('1899-12-30') + pd.to_timedelta(float(value), unit='D')
            except Exception:
                return pd.NaT
        try:
            return pd.to_datetime(value)
        except Exception:
            return pd.NaT

    @staticmethod
    def _match_market_column(columns: List, ticker_aliases: List[str]) -> str:
        """
        Locate the column name in the Market Data sheet that corresponds to any alias.
        """
        if not ticker_aliases:
            return None

        normalised_aliases = [
            alias.strip().upper().replace(' ', '')
            for alias in ticker_aliases
        ]

        for col in columns:
            if not isinstance(col, str):
                continue
            normalised_col = col.strip().upper().replace(' ', '')
            for alias in normalised_aliases:
                if (
                    normalised_col == alias
                    or normalised_col == f"{alias}INDEX"
                    or normalised_col.startswith(alias)
                    or normalised_col.endswith(alias)
                ):
                    return col
        return None

    def calculate_statistics(self) -> Dict:
        """
        Calculate key statistics for all asset classes

        Returns:
            Dictionary containing statistics for each asset class
        """
        if not hasattr(self, 'returns_data'):
            self.load_data()

        stats = {}

        for asset in self.asset_classes:
            if asset in self.returns_data.columns:
                returns = self.returns_data[asset].dropna()

                # Annualized return (geometric mean)
                monthly_returns = 1 + returns
                geometric_mean = monthly_returns.prod() ** (1/len(returns)) - 1
                annualized_return = (1 + geometric_mean) ** 12 - 1

                # Annualized volatility
                annualized_vol = returns.std() * np.sqrt(12)

                # Other statistics
                stats[asset] = {
                    'annualized_return': annualized_return,
                    'annualized_volatility': annualized_vol,
                    'monthly_mean': returns.mean(),
                    'monthly_std': returns.std(),
                    'min_return': returns.min(),
                    'max_return': returns.max(),
                    'skewness': returns.skew(),
                    'kurtosis': returns.kurtosis(),
                    'sharpe_ratio': annualized_return / annualized_vol if annualized_vol > 0 else 0
                }

        return stats

    def get_covariance_matrix(self, annualized: bool = True) -> pd.DataFrame:
        """
        Calculate the covariance matrix of asset returns

        Args:
            annualized: Whether to annualize the covariance matrix

        Returns:
            Covariance matrix as DataFrame
        """
        if not hasattr(self, 'returns_data'):
            self.load_data()

        # Calculate monthly covariance
        cov_matrix = self.returns_data.cov()

        if annualized:
            # Annualize by multiplying by 12
            cov_matrix = cov_matrix * 12

        return cov_matrix

    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        Calculate the correlation matrix of asset returns

        Returns:
            Correlation matrix as DataFrame
        """
        if not hasattr(self, 'returns_data'):
            self.load_data()

        return self.returns_data.corr()

    def get_expected_returns(self, method: str = 'historical') -> pd.Series:
        """
        Calculate expected returns using various methods

        Args:
            method: Method to use ('historical', 'ewma', 'shrinkage')

        Returns:
            Series of expected annual returns
        """
        if not hasattr(self, 'returns_data'):
            self.load_data()

        if method == 'historical':
            # Simple historical average (arithmetic mean for simplicity)
            monthly_means = self.returns_data.mean()
            annual_returns = monthly_means * 12

        elif method == 'ewma':
            # Exponentially weighted moving average
            # Give more weight to recent observations
            lambda_param = 0.94  # Decay factor
            weights = np.array([(1-lambda_param) * lambda_param**i
                              for i in range(len(self.returns_data)-1, -1, -1)])
            weights = weights / weights.sum()

            annual_returns = pd.Series(index=self.returns_data.columns)
            for asset in self.returns_data.columns:
                weighted_returns = self.returns_data[asset].values * weights
                monthly_mean = np.nansum(weighted_returns)
                annual_returns[asset] = monthly_mean * 12

        elif method == 'shrinkage':
            # James-Stein shrinkage toward grand mean
            monthly_means = self.returns_data.mean()
            grand_mean = monthly_means.mean()

            # Shrinkage intensity (simplified)
            shrinkage_factor = 0.3
            shrunk_means = shrinkage_factor * grand_mean + (1 - shrinkage_factor) * monthly_means
            annual_returns = shrunk_means * 12

        else:
            raise ValueError(f"Unknown method: {method}")

        return annual_returns

    def get_rolling_statistics(self, window: int = 36) -> Dict:
        """
        Calculate rolling window statistics

        Args:
            window: Window size in months

        Returns:
            Dictionary with rolling statistics DataFrames
        """
        if not hasattr(self, 'returns_data'):
            self.load_data()

        rolling_mean = self.returns_data.rolling(window=window).mean() * 12
        rolling_std = self.returns_data.rolling(window=window).std() * np.sqrt(12)
        rolling_sharpe = rolling_mean / rolling_std

        return {
            'rolling_mean': rolling_mean,
            'rolling_std': rolling_std,
            'rolling_sharpe': rolling_sharpe
        }

    def split_growth_defensive(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split returns data into growth and defensive assets

        Returns:
            Tuple of (growth_returns, defensive_returns) DataFrames
        """
        if not hasattr(self, 'returns_data'):
            self.load_data()

        growth_returns = self.returns_data[self.growth_assets]
        defensive_returns = self.returns_data[self.defensive_assets]

        return growth_returns, defensive_returns


def test_loader():
    """Test function to verify data loader works correctly"""

    # Initialize loader
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data',
        'BBG Data (2000-2025).xlsx'
    )
    loader = AssetDataLoader(data_path)

    # Load data
    returns_df = loader.load_data()
    print(f"Data shape: {returns_df.shape}")
    print(f"Date range: {returns_df.index[0]} to {returns_df.index[-1]}")
    print(f"\nFirst 5 rows:")
    print(returns_df.head())

    # Calculate statistics
    stats = loader.calculate_statistics()
    print("\nAnnualized Returns:")
    for asset, stat in stats.items():
        print(f"{asset[:30]:30} {stat['annualized_return']:8.2%}")

    # Get covariance matrix
    cov_matrix = loader.get_covariance_matrix()
    print(f"\nCovariance matrix shape: {cov_matrix.shape}")

    # Get expected returns using different methods
    hist_returns = loader.get_expected_returns('historical')
    ewma_returns = loader.get_expected_returns('ewma')
    shrink_returns = loader.get_expected_returns('shrinkage')

    print("\nExpected Returns Comparison:")
    print(f"{'Asset':40} {'Historical':>10} {'EWMA':>10} {'Shrinkage':>10}")
    for asset in hist_returns.index:
        print(f"{asset[:40]:40} {hist_returns[asset]:10.2%} {ewma_returns[asset]:10.2%} {shrink_returns[asset]:10.2%}")

    return loader


if __name__ == "__main__":
    test_loader()
