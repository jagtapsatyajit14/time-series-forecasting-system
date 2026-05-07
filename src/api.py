from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.models.arima_model import ArimaModel
from src.models.prophet_model import ProphetModel
from src.models.xgboost_model import XGBoostModel
from src.models.lstm_model import LSTMModel
from src.model_selection import ModelSelector

# Initialize FastAPI app
app = FastAPI(
    title="Time Series Forecasting API",
    description="API for sales forecasting using multiple ML models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/dashboard")
async def dashboard():
    """Serve interactive dashboard"""
    return FileResponse("../simple_dashboard.html")

@app.get("/")
async def home():
    """Serve simple dashboard as default"""
    return FileResponse("../simple_dashboard.html")

# Global variables for models and data
model_selector = None
processed_data = None
feature_engineer = None

class ForecastRequest(BaseModel):
    states: Optional[List[str]] = None
    weeks_ahead: int = 8
    model_name: Optional[str] = None

class ForecastResponse(BaseModel):
    state: str
    model_used: str
    forecast: List[Dict[str, Any]]
    metrics: Dict[str, float]

class ModelInfo(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    models_loaded: bool
    data_loaded: bool

@app.on_event("startup")
async def startup_event():
    """Initialize models and data on startup"""
    global model_selector, processed_data, feature_engineer
    
    try:
        # Load and process data
        preprocessor = DataPreprocessor()
        data_path = "data/Forecasting Case- Study.xlsx"
        
        if os.path.exists(data_path):
            processed_data = preprocessor.preprocess_data(data_path)
            
            # Feature engineering
            feature_engineer = FeatureEngineer()
            processed_data = feature_engineer.engineer_features(processed_data)
            
            print("Data loaded and processed successfully")
        else:
            print(f"Data file not found: {data_path}")
            
    except Exception as e:
        print(f"Error during startup: {e}")

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        models_loaded=model_selector is not None,
        data_loaded=processed_data is not None
    )

@app.get("/models", response_model=List[ModelInfo])
async def get_available_models():
    """Get list of available forecasting models"""
    models = [
        ModelInfo(
            name="ARIMA",
            description="AutoRegressive Integrated Moving Average model",
            parameters={"order": "Auto-selected", "seasonal": False}
        ),
        ModelInfo(
            name="SARIMA",
            description="Seasonal AutoRegressive Integrated Moving Average model",
            parameters={"order": "Auto-selected", "seasonal": True}
        ),
        ModelInfo(
            name="Prophet",
            description="Facebook's Prophet forecasting model",
            parameters={"seasonality": True, "holidays": True}
        ),
        ModelInfo(
            name="XGBoost",
            description="Gradient boosting model with lag features",
            parameters={"n_estimators": 200, "max_depth": 6}
        ),
        ModelInfo(
            name="LSTM",
            description="Long Short-Term Memory neural network",
            parameters={"sequence_length": 30, "epochs": 50}
        )
    ]
    return models

@app.post("/train-models")
async def train_models():
    """Train all forecasting models"""
    global model_selector, processed_data
    
    if processed_data is None:
        raise HTTPException(status_code=400, detail="Data not loaded")
    
    try:
        # Initialize models
        arima = ArimaModel(model_type='arima')
        sarima = ArimaModel(model_type='sarima')
        prophet = ProphetModel()
        xgboost = XGBoostModel()
        lstm = LSTMModel()
        
        # Train models
        print("Training ARIMA model...")
        arima_models = arima.train(processed_data)
        arima_eval = arima.evaluate(processed_data)
        
        print("Training SARIMA model...")
        sarima_models = sarima.train(processed_data)
        sarima_eval = sarima.evaluate(processed_data)
        
        print("Training Prophet model...")
        prophet_models = prophet.train(processed_data)
        prophet_eval = prophet.evaluate(processed_data)
        
        print("Training XGBoost model...")
        xgboost_models = xgboost.train(processed_data)
        xgboost_eval = xgboost.evaluate(processed_data)
        
        print("Training LSTM model...")
        lstm_models = lstm.train(processed_data)
        lstm_eval = lstm.evaluate(processed_data)
        
        # Model selection
        model_selector = ModelSelector()
        model_selector.add_model('ARIMA', arima, arima_eval)
        model_selector.add_model('SARIMA', sarima, sarima_eval)
        model_selector.add_model('Prophet', prophet, prophet_eval)
        model_selector.add_model('XGBoost', xgboost, xgboost_eval)
        model_selector.add_model('LSTM', lstm, lstm_eval)
        
        # Compare models and select best
        comparison_results, best_models = model_selector.compare_models()
        
        # Save best models
        model_selector.save_best_models()
        
        return {
            "status": "success",
            "message": "Models trained and compared successfully",
            "best_models": best_models.to_dict('records') if best_models is not None else [],
            "comparison_results": comparison_results.to_dict('records') if comparison_results is not None else []
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training models: {str(e)}")

@app.post("/predict", response_model=List[ForecastResponse])
async def predict_sales(request: ForecastRequest):
    """Get sales forecasts for specified states"""
    global model_selector, processed_data
    
    if model_selector is None:
        raise HTTPException(status_code=400, detail="Models not trained. Call /train-models first.")
    
    if processed_data is None:
        raise HTTPException(status_code=400, detail="Data not loaded")
    
    try:
        # Determine which states to predict
        if request.states:
            states_to_predict = request.states
        else:
            states_to_predict = processed_data['State'].unique().tolist()
        
        # Generate forecasts
        forecast_periods = request.weeks_ahead * 7  # Convert weeks to days
        forecasts = []
        
        for state in states_to_predict:
            # Get best model for this state
            best_model_info = model_selector.get_best_model_for_state(state)
            
            if best_model_info is None:
                continue
            
            model_name = best_model_info['model_name']
            model_object = best_model_info['model_object']
            metrics = best_model_info['metrics']
            
            # Get predictions
            if hasattr(model_object, 'predict'):
                predictions = model_object.predict(processed_data, forecast_periods)
                
                if state in predictions:
                    forecast_df = predictions[state]
                    
                    # Convert to response format
                    forecast_list = []
                    for _, row in forecast_df.iterrows():
                        forecast_list.append({
                            "date": row['Date'].isoformat(),
                            "predicted_sales": float(row['Predicted_Sales'])
                        })
                    
                    forecasts.append(ForecastResponse(
                        state=state,
                        model_used=model_name,
                        forecast=forecast_list,
                        metrics=metrics
                    ))
        
        return forecasts
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating predictions: {str(e)}")

@app.get("/model-performance")
async def get_model_performance():
    """Get model performance comparison"""
    global model_selector
    
    if model_selector is None:
        raise HTTPException(status_code=400, detail="Models not trained")
    
    try:
        ranking = model_selector.get_model_ranking()
        summary = model_selector.get_model_summary()
        
        return {
            "model_ranking": ranking.to_dict() if ranking is not None else {},
            "summary": summary
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model performance: {str(e)}")

@app.get("/states")
async def get_available_states():
    """Get list of available states in the dataset"""
    global processed_data
    
    if processed_data is None:
        raise HTTPException(status_code=400, detail="Data not loaded")
    
    states = processed_data['State'].unique().tolist()
    return {"states": states}

@app.get("/data-info")
async def get_data_info():
    """Get information about the dataset"""
    global processed_data
    
    if processed_data is None:
        raise HTTPException(status_code=400, detail="Data not loaded")
    
    info = {
        "total_records": len(processed_data),
        "date_range": {
            "start": processed_data['Date'].min().isoformat(),
            "end": processed_data['Date'].max().isoformat()
        },
        "states": processed_data['State'].unique().tolist(),
        "num_states": processed_data['State'].nunique(),
        "columns": list(processed_data.columns),
        "missing_values": processed_data.isnull().sum().to_dict()
    }
    
    return info

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
