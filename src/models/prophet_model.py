import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class ProphetModel:
    def __init__(self):
        self.models = {}
        self.holiday_data = None
        
    def prepare_prophet_data(self, df, date_col='Date', value_col='Sales'):
        """Prepare data in Prophet format"""
        prophet_df = df.rename(columns={
            date_col: 'ds',
            value_col: 'y'
        })
        return prophet_df[['ds', 'y']]
    
    def add_holidays_to_model(self, model, country='US'):
        """Add holidays to Prophet model"""
        try:
            model.add_country_holidays(country_name=country)
        except Exception as e:
            print(f"Could not add holidays: {e}")
        return model
    
    def train(self, df, date_col='Date', state_col='State', value_col='Total'):
        """Train Prophet models for each state"""
        print("Training Prophet models...")
        
        states = df[state_col].unique()
        
        for state in states:
            print(f"Training Prophet model for {state}...")
            
            try:
                # Get state data
                state_data = df[df[state_col] == state].copy()
                state_data = state_data.sort_values(date_col)
                
                # Prepare data for Prophet
                prophet_data = self.prepare_prophet_data(state_data, date_col, value_col)
                
                # Create and configure Prophet model
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    changepoint_prior_scale=0.05,
                    seasonality_prior_scale=10,
                    holidays_prior_scale=10,
                    mcmc_samples=0,
                    interval_width=0.8,
                    uncertainty_samples=1000
                )
                
                # Add holidays
                model = self.add_holidays_to_model(model)
                
                # Add custom seasonalities if needed
                if len(prophet_data) >= 365:
                    model.add_seasonality(name='monthly', period=30.5, fourier_order=8)
                
                # Fit model
                model.fit(prophet_data)
                
                self.models[state] = model
                print(f"  Model trained successfully for {state}")
                
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
                    
                    # Create future dataframe
                    future = self.models[state].make_future_dataframe(periods=forecast_periods, freq='D')
                    
                    # Make predictions
                    forecast = self.models[state].predict(future)
                    
                    # Extract only future predictions
                    future_predictions = forecast[forecast['ds'] > last_date]
                    
                    # Create predictions dataframe
                    predictions_df = pd.DataFrame({
                        'Date': future_predictions['ds'],
                        'State': state,
                        'Predicted_Sales': future_predictions['yhat'],
                        'Lower_Bound': future_predictions['yhat_lower'],
                        'Upper_Bound': future_predictions['yhat_upper']
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
                    
                    # Prepare training data for Prophet
                    train_prophet = self.prepare_prophet_data(train_data, date_col, value_col)
                    
                    # Train model on training data
                    model = Prophet(
                        yearly_seasonality=True,
                        weekly_seasonality=True,
                        daily_seasonality=False
                    )
                    model = self.add_holidays_to_model(model)
                    model.fit(train_prophet)
                    
                    # Create future dataframe for test period
                    future = model.make_future_dataframe(periods=test_size, freq='D')
                    
                    # Make predictions
                    forecast = model.predict(future)
                    
                    # Extract test period predictions
                    test_predictions = forecast[-test_size:]['yhat'].values
                    
                    # Calculate metrics
                    mae = mean_absolute_error(test_data[value_col], test_predictions)
                    mse = mean_squared_error(test_data[value_col], test_predictions)
                    rmse = np.sqrt(mse)
                    mape = np.mean(np.abs((test_data[value_col] - test_predictions) / test_data[value_col])) * 100
                    
                    results.append({
                        'State': state,
                        'Model': 'Prophet',
                        'MAE': mae,
                        'MSE': mse,
                        'RMSE': rmse,
                        'MAPE': mape
                    })
                    
                except Exception as e:
                    print(f"Error evaluating {state}: {e}")
                    continue
        
        return pd.DataFrame(results)
    
    def get_forecast_components(self, state, forecast_periods=56):
        """Get forecast components (trend, seasonality, holidays)"""
        if state not in self.models:
            return None
        
        try:
            # Create future dataframe
            future = self.models[state].make_future_dataframe(periods=forecast_periods, freq='D')
            
            # Make predictions
            forecast = self.models[state].predict(future)
            
            return forecast[['ds', 'yhat', 'trend', 'seasonal', 'holidays']]
            
        except Exception as e:
            print(f"Error getting components for {state}: {e}")
            return None
