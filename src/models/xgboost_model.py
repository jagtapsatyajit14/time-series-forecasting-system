import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

class XGBoostModel:
    def __init__(self):
        self.models = {}
        self.feature_columns = {}
        
    def prepare_features(self, df, date_col='Date', state_col='State', value_col='Sales'):
        """Prepare features for XGBoost model"""
        df = df.copy()
        
        # Convert date to datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Sort by state and date
        df = df.sort_values([state_col, date_col])
        
        # Identify feature columns (exclude date, state, and target)
        exclude_cols = [date_col, state_col, value_col]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        return df, feature_cols
    
    def train_single_model(self, train_data, feature_cols, target_col='Sales'):
        """Train XGBoost model for a single state"""
        try:
            # Prepare training data
            X_train = train_data[feature_cols]
            y_train = train_data[target_col]
            
            # Create XGBoost model with parameters optimized for time series
            model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                objective='reg:squarederror',
                n_jobs=-1
            )
            
            # Train model
            model.fit(X_train, y_train)
            
            return model
            
        except Exception as e:
            print(f"Error training XGBoost model: {e}")
            return None
    
    def train(self, df, date_col='Date', state_col='State', value_col='Total', test_size=56):
        """Train XGBoost models for each state"""
        print("Training XGBoost models...")
        
        # Prepare features
        df, feature_cols = self.prepare_features(df, date_col, state_col, value_col)
        
        states = df[state_col].unique()
        
        for state in states:
            print(f"Training XGBoost model for {state}...")
            
            try:
                # Get state data
                state_data = df[df[state_col] == state].copy()
                
                # Split data (time series split)
                train_data = state_data[:-test_size]
                
                if len(train_data) < 100:  # Minimum data requirement
                    print(f"  Insufficient data for {state}. Skipping...")
                    continue
                
                # Train model
                model = self.train_single_model(train_data, feature_cols, value_col)
                
                if model is not None:
                    self.models[state] = model
                    self.feature_columns[state] = feature_cols
                    print(f"  Model trained successfully for {state}")
                
            except Exception as e:
                print(f"  Error training model for {state}: {e}")
                continue
        
        print(f"Training completed. Models trained for {len(self.models)} states")
        return self.models
    
    def predict(self, df, forecast_periods=56, date_col='Date', state_col='State', value_col='Total'):
        """Make predictions for next forecast_periods days"""
        predictions = {}
        
        for state in df[state_col].unique():
            if state in self.models:
                try:
                    # Get state data
                    state_data = df[df[state_col] == state].copy()
                    state_data = state_data.sort_values(date_col)
                    
                    # Get feature columns
                    feature_cols = self.feature_columns[state]
                    
                    # Create future dataframe
                    last_date = state_data[date_col].max()
                    future_dates = pd.date_range(
                        start=last_date + pd.Timedelta(days=1),
                        periods=forecast_periods,
                        freq='D'
                    )
                    
                    future_df = pd.DataFrame({
                        date_col: future_dates,
                        state_col: state
                    })
                    
                    # For simplicity, we'll use the last known values for lag features
                    # In a production system, you would iteratively update these
                    last_row = state_data.iloc[-1].copy()
                    
                    future_predictions = []
                    for i, date in enumerate(future_dates):
                        # Create feature row for this date
                        feature_row = last_row.copy()
                        feature_row[date_col] = date
                        
                        # Update time-based features
                        feature_row['year'] = date.year
                        feature_row['month'] = date.month
                        feature_row['day'] = date.day
                        feature_row['dayofweek'] = date.dayofweek
                        feature_row['dayofyear'] = date.dayofyear
                        feature_row['week'] = date.isocalendar().week
                        feature_row['quarter'] = date.quarter
                        
                        # Update cyclical features
                        feature_row['month_sin'] = np.sin(2 * np.pi * date.month / 12)
                        feature_row['month_cos'] = np.cos(2 * np.pi * date.month / 12)
                        feature_row['dayofweek_sin'] = np.sin(2 * np.pi * date.dayofweek / 7)
                        feature_row['dayofweek_cos'] = np.cos(2 * np.pi * date.dayofweek / 7)
                        feature_row['is_weekend'] = 1 if date.dayofweek >= 5 else 0
                        
                        # Get prediction
                        X_pred = feature_row[feature_cols].values.reshape(1, -1)
                        prediction = self.models[state].predict(X_pred)[0]
                        
                        future_predictions.append(prediction)
                        
                        # Update lag features for next iteration (using prediction)
                        if i == 0:
                            last_row['lag_1'] = last_row[value_col]
                        else:
                            last_row['lag_1'] = future_predictions[i-1]
                    
                    # Create predictions dataframe
                    predictions_df = pd.DataFrame({
                        'Date': future_dates,
                        'State': state,
                        'Predicted_Sales': future_predictions
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
                    
                    # Get feature columns
                    feature_cols = self.feature_columns[state]
                    
                    # Make predictions on test set
                    X_test = test_data[feature_cols]
                    predictions = self.models[state].predict(X_test)
                    
                    # Calculate metrics
                    mae = mean_absolute_error(test_data[value_col], predictions)
                    mse = mean_squared_error(test_data[value_col], predictions)
                    rmse = np.sqrt(mse)
                    mape = np.mean(np.abs((test_data[value_col] - predictions) / test_data[value_col])) * 100
                    
                    results.append({
                        'State': state,
                        'Model': 'XGBoost',
                        'MAE': mae,
                        'MSE': mse,
                        'RMSE': rmse,
                        'MAPE': mape
                    })
                    
                except Exception as e:
                    print(f"Error evaluating {state}: {e}")
                    continue
        
        return pd.DataFrame(results)
    
    def get_feature_importance(self, state):
        """Get feature importance for a specific state"""
        if state not in self.models:
            return None
        
        try:
            model = self.models[state]
            feature_cols = self.feature_columns[state]
            
            importance = model.feature_importances_
            
            importance_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            return importance_df
            
        except Exception as e:
            print(f"Error getting feature importance for {state}: {e}")
            return None
