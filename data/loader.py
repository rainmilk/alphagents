"""
Data Loader Module

This module handles:
1. Loading stock data from various sources (Tushare, AkShare, local files)
2. Preprocessing (missing values, outliers, normalization)
3. Feature engineering (technical indicators, fundamental features)
4. Train/test splitting

Author: AAAI 2027 LLM Multi-Factor Stock Selection Project
Date: 2026-06-07
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import yaml
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """
    Data loader for stock market data.
    
    Supports multiple data sources and handles preprocessing.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize data loader with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.preprocessing_config = self.data_config['preprocessing']
        
        # Data cache
        self.price_data = None
        self.fundamental_data = None
        self.industry_data = None
        
    def load_data(
        self,
        start_date: str = None,
        end_date: str = None,
        universe: str = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Load stock data for the specified period and universe.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            universe: Stock universe (hs300, zz500, all_a)
            
        Returns:
            Tuple of (price_data, fundamental_data, industry_series)
            - price_data: DataFrame, shape (n_days, n_stocks)
            - fundamental_data: Dict of DataFrames
            - industry_series: Series, index = stock codes
        """
        start_date = start_date or self.data_config['universe']['start_date']
        end_date = end_date or self.data_config['universe']['end_date']
        universe = universe or self.data_config['universe']['index']
        
        print(f"Loading data from {start_date} to {end_date}, universe: {universe}")
        
        # Try to load from local cache first
        cache_path = f"data/cache_{universe}_{start_date}_{end_date}.pkl"
        if os.path.exists(cache_path):
            print(f"Loading from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                return pd.read_pickle(f)
        
        # Load from data source
        if self.data_config['source'] == 'tushare':
            price_data, fundamental_data, industry_data = self._load_from_tushare(
                start_date, end_date, universe
            )
        elif self.data_config['source'] == 'akshare':
            price_data, fundamental_data, industry_data = self._load_from_akshare(
                start_date, end_date, universe
            )
        else:
            raise ValueError(f"Unsupported data source: {self.data_config['source']}")
        
        # Preprocess
        price_data = self._preprocess_price_data(price_data)
        fundamental_data = self._preprocess_fundamental_data(fundamental_data)
        
        # Save to cache
        os.makedirs('data', exist_ok=True)
        with open(cache_path, 'wb') as f:
            pd.to_pickle((price_data, fundamental_data, industry_data), f)
        
        self.price_data = price_data
        self.fundamental_data = fundamental_data
        self.industry_data = industry_data
        
        return price_data, fundamental_data, industry_data
    
    def _load_from_tushare(
        self,
        start_date: str,
        end_date: str,
        universe: str,
    ) -> Tuple[pd.DataFrame, Dict, pd.Series]:
        """
        Load data from Tushare API.
        
        Note: This is a mock implementation. In practice, you need to:
        1. Install tushare: pip install tushare
        2. Set your token: ts.set_token('your_token')
        3. Replace mock data with real API calls
        """
        print("Loading from Tushare (mock data)...")
        
        # Generate mock data for demonstration
        dates = pd.date_range(start_date, end_date, freq='B')  # Business days
        
        # Get stock list
        if universe == 'hs300':
            n_stocks = 300
            stock_codes = [f'STOCK_{i:04d}' for i in range(n_stocks)]
        elif universe == 'zz500':
            n_stocks = 500
            stock_codes = [f'STOCK_{i:04d}' for i in range(n_stocks)]
        else:
            n_stocks = 1000
            stock_codes = [f'STOCK_{i:04d}' for i in range(n_stocks)]
        
        # Generate mock price data
        np.random.seed(42)
        n_days = len(dates)
        
        # Price data (simulated)
        price_data = {}
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            if col in ['open', 'high', 'low', 'close']:
                # Simulate price series with random walk
                base_price = 10 + np.random.randn(n_stocks) * 5
                returns = np.random.randn(n_days, n_stocks) * 0.02
                prices = base_price * np.exp(np.cumsum(returns, axis=0))
                price_data[col] = pd.DataFrame(prices, index=dates, columns=stock_codes)
            else:
                # Volume and amount
                volume = np.abs(np.random.randn(n_days, n_stocks) * 1e6)
                price_data[col] = pd.DataFrame(volume, index=dates, columns=stock_codes)
        
        # Fundamental data (mock)
        fundamental_data = {
            'pe': pd.DataFrame(np.abs(np.random.randn(n_days, n_stocks) * 20 + 15), index=dates, columns=stock_codes),
            'pb': pd.DataFrame(np.abs(np.random.randn(n_days, n_stocks) * 3 + 1), index=dates, columns=stock_codes),
            'roe': pd.DataFrame(np.random.randn(n_days, n_stocks) * 0.1 + 0.1, index=dates, columns=stock_codes),
            'market_cap': pd.DataFrame(np.random.randn(n_days, n_stocks) * 1e9 + 5e9, index=dates, columns=stock_codes),
        }
        
        # Industry data (mock)
        industries = ['Technology', 'Finance', 'Healthcare', 'Consumer', 'Energy', 'Materials', 'Industrial']
        industry_series = pd.Series(
            np.random.choice(industries, size=n_stocks),
            index=stock_codes
        )
        
        return price_data, fundamental_data, industry_series
    
    def _load_from_akshare(
        self,
        start_date: str,
        end_date: str,
        universe: str,
    ) -> Tuple[pd.DataFrame, Dict, pd.Series]:
        """
        Load data from AkShare API.
        
        Note: This is a mock implementation. In practice, you need to:
        1. Install akshare: pip install akshare
        2. Replace mock data with real API calls
        """
        print("Loading from AkShare (mock data)...")
        # Similar to _load_from_tushare, but using AkShare API
        return self._load_from_tushare(start_date, end_date, universe)
    
    def _preprocess_price_data(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Preprocess price data.
        
        Args:
            price_data: Dict of DataFrames (open, high, low, close, volume, amount)
            
        Returns:
            Preprocessed price data
        """
        processed = {}
        
        for col, df in price_data.items():
            # Fill missing values
            if self.preprocessing_config['fill_method'] == 'forward':
                df = df.fillna(method='ffill')
            elif self.preprocessing_config['fill_method'] == 'backward':
                df = df.fillna(method='bfill')
            elif self.preprocessing_config['fill_method'] == 'interpolate':
                df = df.interpolate(method='linear')
            
            # Handle outliers
            if self.preprocessing_config['outlier_method'] == 'winsorize':
                df = self._winsorize(df, self.preprocessing_config['outlier_threshold'])
            elif self.preprocessing_config['outlier_method'] == 'clip':
                df = self._clip_outliers(df, self.preprocessing_config['outlier_threshold'])
            
            processed[col] = df
        
        return processed
    
    def _preprocess_fundamental_data(self, fundamental_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Preprocess fundamental data.
        
        Args:
            fundamental_data: Dict of DataFrames
            
        Returns:
            Preprocessed fundamental data
        """
        processed = {}
        
        for col, df in fundamental_data.items():
            # Fill missing values with forward fill (use latest available)
            df = df.fillna(method='ffill').fillna(method='bfill')
            processed[col] = df
        
        return processed
    
    def _winsorize(self, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
        """
        Winsorize data (clip at percentiles).
        
        Args:
            df: DataFrame to winsorize
            threshold: Percentile threshold (e.g., 3.0 means clip at 0.5% and 99.5%)
            
        Returns:
            Winsorized DataFrame
        """
        lower = threshold / 100
        upper = 1 - threshold / 100
        
        result = df.copy()
        for col in df.columns:
            q_low = df[col].quantile(lower)
            q_high = df[col].quantile(upper)
            result[col] = df[col].clip(lower=q_low, upper=q_high)
        
        return result
    
    def _clip_outliers(self, df: pd.DataFrame, sigma: float) -> pd.DataFrame:
        """
        Clip outliers using sigma rule.
        
        Args:
            df: DataFrame to clip
            sigma: Number of standard deviations
            
        Returns:
            Clipped DataFrame
        """
        result = df.copy()
        for col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            result[col] = df[col].clip(lower=mean - sigma * std, upper=mean + sigma * std)
        
        return result
    
    def get_returns(
        self,
        price_col: str = 'close',
        period: int = 1,
    ) -> pd.DataFrame:
        """
        Calculate stock returns.
        
        Args:
            price_col: Price column to use ('open', 'high', 'low', 'close')
            period: Return period (1 = daily, 5 = weekly, etc.)
            
        Returns:
            DataFrame of returns
        """
        if self.price_data is None:
            raise ValueError("Price data not loaded. Call load_data() first.")
        
        close_prices = self.price_data[price_col]
        returns = close_prices.pct_change(period).shift(-period)
        
        return returns
    
    def get_market_cap(self) -> pd.DataFrame:
        """
        Get market capitalization data.
        
        Returns:
            DataFrame of market cap
        """
        if self.fundamental_data is None:
            raise ValueError("Fundamental data not loaded. Call load_data() first.")
        
        return self.fundamental_data.get('market_cap', pd.DataFrame())
    
    def get_industry(self) -> pd.Series:
        """
        Get industry classification.
        
        Returns:
            Series of industry labels
        """
        if self.industry_data is None:
            raise ValueError("Industry data not loaded. Call load_data() first.")
        
        return self.industry_data
    
    def split_data(
        self,
        test_ratio: float = 0.2,
        method: str = 'chronological',
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets.
        
        Args:
            test_ratio: Ratio of test data
            method: Split method ('chronological', 'rolling')
            
        Returns:
            Tuple of (train_data, test_data)
        """
        if self.price_data is None:
            raise ValueError("Price data not loaded. Call load_data() first.")
        
        close_prices = self.price_data['close']
        n_days = len(close_prices)
        
        if method == 'chronological':
            split_idx = int(n_days * (1 - test_ratio))
            train_data = close_prices.iloc[:split_idx]
            test_data = close_prices.iloc[split_idx:]
        elif method == 'rolling':
            # For rolling window cross-validation
            # This returns the full data, actual splitting is done in experiment runner
            train_data = close_prices
            test_data = close_prices
        else:
            raise ValueError(f"Unsupported split method: {method}")
        
        return train_data, test_data


def load_sample_data(n_stocks: int = 100, n_days: int = 1000) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Load sample data for quick testing.
    
    Args:
        n_stocks: Number of stocks
        n_days: Number of trading days
        
    Returns:
        Tuple of (price_data, fundamental_data, industry_series)
    """
    np.random.seed(42)
    
    # Generate dates
    dates = pd.date_range('2020-01-01', periods=n_days, freq='B')
    stock_codes = [f'STOCK_{i:04d}' for i in range(n_stocks)]
    
    # Price data
    price_data = {}
    for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        if col in ['open', 'high', 'low', 'close']:
            base_price = 10 + np.random.randn(n_stocks) * 5
            returns = np.random.randn(n_days, n_stocks) * 0.02
            prices = base_price * np.exp(np.cumsum(returns, axis=0))
            price_data[col] = pd.DataFrame(prices, index=dates, columns=stock_codes)
        else:
            volume = np.abs(np.random.randn(n_days, n_stocks) * 1e6)
            price_data[col] = pd.DataFrame(volume, index=dates, columns=stock_codes)
    
    # Fundamental data
    fundamental_data = {
        'pe': pd.DataFrame(np.abs(np.random.randn(n_days, n_stocks) * 20 + 15), index=dates, columns=stock_codes),
        'pb': pd.DataFrame(np.abs(np.random.randn(n_days, n_stocks) * 3 + 1), index=dates, columns=stock_codes),
        'roe': pd.DataFrame(np.random.randn(n_days, n_stocks) * 0.1 + 0.1, index=dates, columns=stock_codes),
        'market_cap': pd.DataFrame(np.random.randn(n_days, n_stocks) * 1e9 + 5e9, index=dates, columns=stock_codes),
    }
    
    # Industry data
    industries = ['Technology', 'Finance', 'Healthcare', 'Consumer', 'Energy']
    industry_series = pd.Series(
        np.random.choice(industries, size=n_stocks),
        index=stock_codes
    )
    
    return price_data, fundamental_data, industry_series


if __name__ == '__main__':
    # Demo
    print("=== Data Loader Demo ===\n")
    
    loader = DataLoader()
    price_data, fundamental_data, industry_data = loader.load_data(
        start_date='2022-01-01',
        end_date='2024-12-31',
        universe='hs300',
    )
    
    print(f"Price data keys: {list(price_data.keys())}")
    print(f"Close prices shape: {price_data['close'].shape}")
    print(f"Fundamental data keys: {list(fundamental_data.keys())}")
    print(f"Number of stocks: {len(industry_data)}")
    print(f"Industries: {industry_data.value_counts().to_dict()}")
    
    # Get returns
    returns = loader.get_returns()
    print(f"\nReturns shape: {returns.shape}")
    print(f"Mean return: {returns.mean().mean():.6f}")
    print(f"Return std: {returns.std().mean():.6f}")
    
    print("\n=== Demo Complete ===")
