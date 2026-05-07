import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

class ModelSelector:
    def __init__(self):
        self.models = {}
        self.evaluation_results = {}
        self.best_models = {}
        
    def add_model(self, model_name, model_object, evaluation_df):
        """Add a model and its evaluation results"""
        self.models[model_name] = model_object
        self.evaluation_results[model_name] = evaluation_df
        
    def compare_models(self):
        """Compare all models and select the best one for each state"""
        if not self.evaluation_results:
            print("No models to compare")
            return None
        
        # Combine all evaluation results
        all_results = []
        for model_name, eval_df in self.evaluation_results.items():
            if eval_df is not None and not eval_df.empty:
                eval_df_copy = eval_df.copy()
                eval_df_copy['Model_Name'] = model_name
                all_results.append(eval_df_copy)
        
        if not all_results:
            print("No valid evaluation results")
            return None
        
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # Select best model for each state based on RMSE
        best_models_per_state = combined_results.loc[
            combined_results.groupby('State')['RMSE'].idxmin()
        ]
        
        # Store best models
        for _, row in best_models_per_state.iterrows():
            state = row['State']
            model_name = row['Model_Name']
            self.best_models[state] = {
                'model_name': model_name,
                'model_object': self.models[model_name],
                'metrics': row[['MAE', 'MSE', 'RMSE', 'MAPE']].to_dict()
            }
        
        print("Model comparison completed:")
        print(combined_results.pivot(index='State', columns='Model_Name', values='RMSE'))
        
        print("\nBest models selected:")
        for state, info in self.best_models.items():
            print(f"{state}: {info['model_name']} (RMSE: {info['metrics']['RMSE']:.2f})")
        
        return combined_results, best_models_per_state
    
    def get_best_model_for_state(self, state):
        """Get the best model for a specific state"""
        if state in self.best_models:
            return self.best_models[state]
        else:
            return None
    
    def get_model_ranking(self):
        """Get overall model ranking across all states"""
        if not self.evaluation_results:
            return None
        
        # Combine all evaluation results
        all_results = []
        for model_name, eval_df in self.evaluation_results.items():
            if eval_df is not None and not eval_df.empty:
                eval_df_copy = eval_df.copy()
                eval_df_copy['Model_Name'] = model_name
                all_results.append(eval_df_copy)
        
        if not all_results:
            return None
        
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # Calculate average metrics per model
        model_averages = combined_results.groupby('Model_Name').agg({
            'MAE': 'mean',
            'MSE': 'mean',
            'RMSE': 'mean',
            'MAPE': 'mean'
        }).sort_values('RMSE')
        
        return model_averages
    
    def save_best_models(self, save_dir='models/saved_models'):
        """Save the best models for each state"""
        os.makedirs(save_dir, exist_ok=True)
        
        for state, info in self.best_models.items():
            model_name = info['model_name']
            model_object = info['model_object']
            
            # Create filename
            filename = f"{state}_{model_name}_best.pkl"
            filepath = os.path.join(save_dir, filename)
            
            try:
                # Save model
                joblib.dump(model_object, filepath)
                print(f"Saved best model for {state}: {filename}")
            except Exception as e:
                print(f"Error saving model for {state}: {e}")
    
    def load_best_models(self, save_dir='models/saved_models'):
        """Load the best models for each state"""
        if not os.path.exists(save_dir):
            print(f"Models directory {save_dir} does not exist")
            return
        
        loaded_models = {}
        
        for filename in os.listdir(save_dir):
            if filename.endswith('.pkl'):
                try:
                    # Extract state and model name from filename
                    parts = filename.replace('.pkl', '').split('_')
                    state = '_'.join(parts[:-2])  # Handle state names with underscores
                    model_name = parts[-2]
                    
                    # Load model
                    filepath = os.path.join(save_dir, filename)
                    model = joblib.load(filepath)
                    
                    loaded_models[state] = {
                        'model_name': model_name,
                        'model_object': model
                    }
                    
                    print(f"Loaded model for {state}: {filename}")
                    
                except Exception as e:
                    print(f"Error loading model {filename}: {e}")
        
        self.best_models = loaded_models
        return loaded_models
    
    def generate_forecast_report(self, df, forecast_periods=56):
        """Generate a comprehensive forecast report"""
        if not self.best_models:
            print("No best models available for forecasting")
            return None
        
        all_forecasts = []
        
        for state, info in self.best_models.items():
            model_name = info['model_name']
            model_object = info['model_object']
            
            try:
                # Get predictions from the model
                if hasattr(model_object, 'predict'):
                    predictions = model_object.predict(df, forecast_periods)
                    
                    if state in predictions:
                        forecast_df = predictions[state]
                        forecast_df['Model'] = model_name
                        all_forecasts.append(forecast_df)
                        
            except Exception as e:
                print(f"Error generating forecast for {state}: {e}")
                continue
        
        if all_forecasts:
            combined_forecasts = pd.concat(all_forecasts, ignore_index=True)
            return combined_forecasts
        else:
            return None
    
    def get_model_summary(self):
        """Get a summary of all models and their performance"""
        summary = {
            'total_states': len(self.best_models),
            'models_available': list(self.models.keys()),
            'best_models_per_state': {
                state: info['model_name'] for state, info in self.best_models.items()
            }
        }
        
        # Add model ranking if available
        ranking = self.get_model_ranking()
        if ranking is not None:
            summary['overall_ranking'] = ranking.to_dict()
        
        return summary
