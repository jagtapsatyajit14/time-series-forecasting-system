import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')

class LSTMModel:
    def __init__(self, sequence_length=30):
        self.models = {}
        self.scalers = {}
        self.sequence_length = sequence_length
        self.feature_columns = {}
        
    def create_sequences(self, data, sequence_length):
        """Create sequences for LSTM training"""
        sequences = []
        targets = []
        
        for i in range(len(data) - sequence_length):
            sequences.append(data[i:(i + sequence_length)])
            targets.append(data[i + sequence_length])
        
        return np.array(sequences), np.array(targets)
    
    def prepare_lstm_data(self, df, date_col='Date', state_col='State', value_col='Sales'):
        """Prepare data for LSTM model"""
        df = df.copy()
        
        # Convert date to datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Sort by state and date
        df = df.sort_values([state_col, date_col])
        
        # Identify feature columns (exclude date, state, and target)
        exclude_cols = [date_col, state_col, value_col]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        return df, feature_cols
    
    def create_lstm_model(self, input_shape):
        """Create LSTM model architecture"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def train_single_model(self, train_data, feature_cols, target_col='Sales'):
        """Train LSTM model for a single state"""
        try:
            # Scale the data
            scaler = MinMaxScaler()
            
            # Prepare features and target
            features = train_data[feature_cols].values
            target = train_data[target_col].values.reshape(-1, 1)
            
            # Scale features and target
            scaled_features = scaler.fit_transform(features)
            scaled_target = scaler.fit_transform(target)
            
            # Create sequences
            X, y = self.create_sequences(scaled_features, self.sequence_length)
            
            if len(X) < 10:  # Minimum sequences requirement
                return None, None
            
            # Create model
            model = self.create_lstm_model((self.sequence_length, len(feature_cols)))
            
            # Train model
            history = model.fit(
                X, y,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
            
            return model, scaler
            
        except Exception as e:
            print(f"Error training LSTM model: {e}")
            return None, None
    
    def train(self, df, date_col='Date', state_col='State', value_col='Total', test_size=56):
        """Train LSTM models for each state"""
        print("Training LSTM models...")
        
        # Prepare data
        df, feature_cols = self.prepare_lstm_data(df, date_col, state_col, value_col)
        
        states = df[state_col].unique()
        
        for state in states:
            print(f"Training LSTM model for {state}...")
            
            try:
                # Get state data
                state_data = df[df[state_col] == state].copy()
                
                # Split data
                train_data = state_data[:-test_size]
                
                if len(train_data) < self.sequence_length + 50:  # Minimum data requirement
                    print(f"  Insufficient data for {state}. Skipping...")
                    continue
                
                # Train model
                model, scaler = self.train_single_model(train_data, feature_cols, value_col)
                
                if model is not None and scaler is not None:
                    self.models[state] = model
                    self.scalers[state] = scaler
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
                    
                    # Get feature columns and scaler
                    feature_cols = self.feature_columns[state]
                    scaler = self.scalers[state]
                    model = self.models[state]
                    
                    # Get the last sequence_length days of data
                    last_data = state_data.tail(self.sequence_length)[feature_cols].values
                    scaled_last_data = scaler.transform(last_data)
                    
                    future_predictions = []
                    current_sequence = scaled_last_data.copy()
                    
                    # Make predictions iteratively
                    for i in range(forecast_periods):
                        # Reshape sequence for prediction
                        X_pred = current_sequence.reshape(1, self.sequence_length, len(feature_cols))
                        
                        # Make prediction
                        scaled_pred = model.predict(X_pred, verbose=0)
                        
                        # Inverse transform to get actual value
                        pred_value = scaler.inverse_transform(scaled_pred)[0][0]
                        future_predictions.append(pred_value)
                        
                        # Update sequence for next prediction
                        # Create a new row with the prediction (simplified approach)
                        new_row = current_sequence[-1].copy()
                        new_row[0] = scaled_pred[0][0]  # Update target column with prediction
                        
                        # Shift sequence and add new row
                        current_sequence = np.vstack([current_sequence[1:], new_row])
                    
                    # Create future dates
                    last_date = state_data[date_col].max()
                    future_dates = pd.date_range(
                        start=last_date + pd.Timedelta(days=1),
                        periods=forecast_periods,
                        freq='D'
                    )
                    
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
                    
                    # Get feature columns and scaler
                    feature_cols = self.feature_columns[state]
                    scaler = self.scalers[state]
                    model = self.models[state]
                    
                    # Prepare test data
                    test_features = test_data[feature_cols].values
                    scaled_test_features = scaler.transform(test_features)
                    
                    # Create sequences for test data
                    X_test, y_test = self.create_sequences(scaled_test_features, self.sequence_length)
                    
                    if len(X_test) == 0:
                        continue
                    
                    # Make predictions
                    scaled_predictions = model.predict(X_test, verbose=0)
                    
                    # Inverse transform predictions and targets
                    predictions = scaler.inverse_transform(scaled_predictions)
                    actual_values = scaler.inverse_transform(y_test.reshape(-1, 1))
                    
                    # Calculate metrics
                    mae = mean_absolute_error(actual_values, predictions)
                    mse = mean_squared_error(actual_values, predictions)
                    rmse = np.sqrt(mse)
                    mape = np.mean(np.abs((actual_values.flatten() - predictions.flatten()) / actual_values.flatten())) * 100
                    
                    results.append({
                        'State': state,
                        'Model': 'LSTM',
                        'MAE': mae,
                        'MSE': mse,
                        'RMSE': rmse,
                        'MAPE': mape
                    })
                    
                except Exception as e:
                    print(f"Error evaluating {state}: {e}")
                    continue
        
        return pd.DataFrame(results)
