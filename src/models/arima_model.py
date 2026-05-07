import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class ArimaModel:
    def __init__(self, model_type='arima'):
        self.model_type = model_type
        self.models = {}
        self.best_params = {}
        
    def find_best_arima_params(self, series, max_p=3, max_d=2, max_q=3):
        """Find best ARIMA parameters using grid search"""
        best_aic = float('inf')
        best_params = None
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        results = model.fit()
                        
                        if results.aic < best_aic:
                            best_aic = results.aic
                            best_params = (p, d, q)
                    except:
                        continue
        
        return best_params
    
    def find_best_sarima_params(self, series, seasonal_period=7):
        """Find best SARIMA parameters"""
        best_aic = float('inf')
        best_params = None
        
        # Common SARIMA parameter combinations
        param_combinations = [
            ((1, 1, 1), (1, 1, 1, seasonal_period)),
            ((1, 1, 1), (0, 1, 1, seasonal_period)),
            ((0, 1, 1), (0, 1, 1, seasonal_period)),
            ((1, 1, 0), (1, 1, 0, seasonal_period)),
        ]
        
        for order, seasonal_order in param_combinations:
            try:
                model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
                results = model.fit()
                
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_params = (order, seasonal_order)
            except:
                continue
        
        return best_params
    
    def train(self, df, date_col='Date', state_col='State', value_col='Total'):
        """Train ARIMA/SARIMA models for each state"""
        print(f"Training {self.model_type.upper()} models...")
        
        states = df[state_col].unique()
        
        for state in states:
            print(f"Training model for {state}...")
            
            # Get state data
            state_data = df[df[state_col] == state].copy()
            state_data = state_data.sort_values(date_col)
            
            # Set date as index
            series = state_data.set_index(date_col)[value_col]
            
            try:
                if self.model_type == 'arima':
                    # Find best parameters
                    best_params = self.find_best_arima_params(series)
                    if best_params:
                        # Train model with best parameters
                        model = ARIMA(series, order=best_params)
                        fitted_model = model.fit()
                        
                        self.models[state] = fitted_model
                        self.best_params[state] = best_params
                        print(f"  Best params for {state}: {best_params}")
                    
                elif self.model_type == 'sarima':
                    # Find best parameters
                    best_params = self.find_best_sarima_params(series)
                    if best_params:
                        order, seasonal_order = best_params
                        # Train model
                        model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
                        fitted_model = model.fit()
                        
                        self.models[state] = fitted_model
                        self.best_params[state] = best_params
                        print(f"  Best params for {state}: {best_params}")
                        
            except Exception as e:
                print(f"  Error training model for {state}: {e}")
                continue
        
        print(f"Training completed. Models trained for {len(self.models)} states")
        return self.models
    
    def predict(self, df, forecast_periods=56, date_col='Date', state_col='State'):
        """Make predictions for next forecast_periods days"""
        predictions = {}
        
        for state in df[state_col].unique():
            if state in self.models:
                try:
                    # Get last date from data
                    last_date = df[df[state_col] == state][date_col].max()
                    
                    # Make forecast
                    forecast = self.models[state].forecast(steps=forecast_periods)
                    
                    # Create date range for forecast
                    forecast_dates = pd.date_range(
                        start=last_date + pd.Timedelta(days=1),
                        periods=forecast_periods,
                        freq='D'
                    )
                    
                    # Create predictions dataframe
                    predictions_df = pd.DataFrame({
                        'Date': forecast_dates,
                        'State': state,
                        'Predicted_Sales': forecast.values
                    })
                    
                    predictions[state] = predictions_df
                    
                except Exception as e:
                    print(f"Error predicting for {state}: {e}")
                    continue
        
        return predictions
    
    def evaluate(self, df, date_col='Date', state_col='State', value_col='Total', test_size=56):
        """Evaluate model performance"""
        results = []
        
        for state in df[state_col].unique():
            if state in self.models:
                try:
                    # Get state data
                    state_data = df[df[state_col] == state].copy()
                    state_data = state_data.sort_values(date_col)
                    
                    # Split data
                    train_data = state_data[:-test_size]
                    test_data = state_data[-test_size:]
                    
                    # Make predictions on test set
                    predictions = self.models[state].forecast(steps=test_size)
                    
                    # Calculate metrics
                    mae = mean_absolute_error(test_data[value_col], predictions)
                    mse = mean_squared_error(test_data[value_col], predictions)
                    rmse = np.sqrt(mse)
                    mape = np.mean(np.abs((test_data[value_col] - predictions) / test_data[value_col])) * 100
                    
                    results.append({
                        'State': state,
                        'Model': self.model_type.upper(),
                        'MAE': mae,
                        'MSE': mse,
                        'RMSE': rmse,
                        'MAPE': mape
                    })
                    
                except Exception as e:
                    print(f"Error evaluating {state}: {e}")
                    continue
        
        return pd.DataFrame(results)
