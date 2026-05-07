import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import holidays

class DataPreprocessor:
    def __init__(self):
        self.us_holidays = holidays.US()
        
    def load_data(self, file_path):
        """Load data from Excel file"""
        try:
            df = pd.read_excel(file_path)
            print(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        # Check for missing values
        missing_info = df.isnull().sum()
        print("Missing values per column:")
        print(missing_info[missing_info > 0])
        
        # Forward fill for time series data
        df = df.ffill()
        
        # If still missing, use backward fill
        df = df.bfill()
        
        return df
    
    def handle_missing_dates(self, df, date_col='Date', state_col='State'):
        """Handle missing dates in time series data"""
        # Convert date column to datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Get unique states
        states = df[state_col].unique()
        
        # Create complete date range for each state
        complete_dfs = []
        
        for state in states:
            state_data = df[df[state_col] == state].copy()
            
            # Get min and max dates for this state
            min_date = state_data[date_col].min()
            max_date = state_data[date_col].max()
            
            # Create complete date range
            date_range = pd.date_range(start=min_date, end=max_date, freq='D')
            
            # Create complete dataframe for this state
            complete_df = pd.DataFrame({date_col: date_range})
            complete_df[state_col] = state
            
            # Merge with original data
            state_complete = complete_df.merge(state_data, on=[date_col, state_col], how='left')
            
            complete_dfs.append(state_complete)
        
        # Combine all states
        df_complete = pd.concat(complete_dfs, ignore_index=True)
        
        # Handle missing values after date completion
        df_complete = self.handle_missing_values(df_complete)
        
        return df_complete
    
    def add_holiday_flags(self, df, date_col='Date'):
        """Add holiday flags to the dataset"""
        df['is_holiday'] = df[date_col].apply(lambda x: x in self.us_holidays)
        df['is_holiday'] = df['is_holiday'].astype(int)
        
        # Add day before and after holiday flags
        df['is_day_before_holiday'] = df[date_col].apply(
            lambda x: (x + timedelta(days=1)) in self.us_holidays
        ).astype(int)
        
        df['is_day_after_holiday'] = df[date_col].apply(
            lambda x: (x - timedelta(days=1)) in self.us_holidays
        ).astype(int)
        
        return df
    
    def preprocess_data(self, file_path, date_col='Date', state_col='State', value_col='Total'):
        """Main preprocessing pipeline"""
        # Load data
        df = self.load_data(file_path)
        if df is None:
            return None
        
        # Handle missing dates
        df = self.handle_missing_dates(df, date_col, state_col)
        
        # Add holiday flags
        df = self.add_holiday_flags(df, date_col)
        
        # Sort by date and state
        df = df.sort_values([state_col, date_col]).reset_index(drop=True)
        
        print(f"Preprocessing completed. Final shape: {df.shape}")
        return df
