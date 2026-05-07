import pandas as pd
import numpy as np
from datetime import datetime

class FeatureEngineer:
    def __init__(self):
        pass
    
    def create_lag_features(self, df, value_col='Total', lags=[1, 7, 30]):
        """Create lag features for time series"""
        df = df.copy()
        
        for lag in lags:
            df[f'lag_{lag}'] = df.groupby('State')[value_col].shift(lag)
        
        return df
    
    def create_rolling_features(self, df, value_col='Total', windows=[7, 14, 30]):
        """Create rolling mean and std features"""
        df = df.copy()
        
        for window in windows:
            df[f'rolling_mean_{window}'] = df.groupby('State')[value_col].transform(
                lambda x: x.rolling(window=window).mean()
            )
            df[f'rolling_std_{window}'] = df.groupby('State')[value_col].transform(
                lambda x: x.rolling(window=window).std()
            )
        
        return df
    
    def create_time_features(self, df, date_col='Date'):
        """Create time-based features"""
        df = df.copy()
        
        # Basic time features
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['day'] = df[date_col].dt.day
        df['dayofweek'] = df[date_col].dt.dayofweek
        df['dayofyear'] = df[date_col].dt.dayofyear
        df['week'] = df[date_col].dt.isocalendar().week
        df['quarter'] = df[date_col].dt.quarter
        
        # Cyclical features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        
        # Weekend flag
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        
        return df
    
    def create_seasonal_features(self, df, date_col='Date', value_col='Total'):
        """Create seasonal decomposition features"""
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        df = df.copy()
        
        # For each state, decompose the time series
        states = df['State'].unique()
        
        for state in states:
            state_data = df[df['State'] == state].copy().sort_values(date_col)
            
            if len(state_data) >= 365:  # Need enough data for decomposition
                try:
                    # Set date as index for decomposition
                    state_data_indexed = state_data.set_index(date_col)
                    
                    # Seasonal decomposition
                    decomposition = seasonal_decompose(
                        state_data_indexed[value_col], 
                        model='additive', 
                        period=365
                    )
                    
                    # Add decomposition components back to dataframe
                    df.loc[df['State'] == state, 'trend'] = decomposition.trend.values
                    df.loc[df['State'] == state, 'seasonal'] = decomposition.seasonal.values
                    df.loc[df['State'] == state, 'residual'] = decomposition.resid.values
                    
                except Exception as e:
                    print(f"Seasonal decomposition failed for state {state}: {e}")
                    # Set default values if decomposition fails
                    df.loc[df['State'] == state, 'trend'] = 0
                    df.loc[df['State'] == state, 'seasonal'] = 0
                    df.loc[df['State'] == state, 'residual'] = 0
            else:
                # Not enough data for decomposition
                df.loc[df['State'] == state, 'trend'] = 0
                df.loc[df['State'] == state, 'seasonal'] = 0
                df.loc[df['State'] == state, 'residual'] = 0
        
        return df
    
    def create_interaction_features(self, df):
        """Create interaction features"""
        df = df.copy()
        
        # Holiday interactions
        df['holiday_weekend'] = df['is_holiday'] * df['is_weekend']
        
        # Lag interactions
        if 'lag_1' in df.columns and 'lag_7' in df.columns:
            df['lag_ratio_1_7'] = df['lag_1'] / (df['lag_7'] + 1e-8)
        
        # Rolling features interactions
        if 'rolling_mean_7' in df.columns and 'rolling_mean_30' in df.columns:
            df['rolling_ratio_7_30'] = df['rolling_mean_7'] / (df['rolling_mean_30'] + 1e-8)
        
        return df
    
    def engineer_features(self, df, date_col='Date', value_col='Total'):
        """Main feature engineering pipeline"""
        print("Starting feature engineering...")
        
        # Create time features
        df = self.create_time_features(df, date_col)
        print("Time features created")
        
        # Create lag features
        df = self.create_lag_features(df, value_col)
        print("Lag features created")
        
        # Create rolling features
        df = self.create_rolling_features(df, value_col)
        print("Rolling features created")
        
        # Create seasonal features
        df = self.create_seasonal_features(df, date_col, value_col)
        print("Seasonal features created")
        
        # Create interaction features
        df = self.create_interaction_features(df)
        print("Interaction features created")
        
        # Remove rows with NaN values (created by lag/rolling features)
        initial_rows = len(df)
        df = df.dropna()
        final_rows = len(df)
        
        print(f"Feature engineering completed. Removed {initial_rows - final_rows} rows with NaN values")
        print(f"Final dataset shape: {df.shape}")
        
        return df
