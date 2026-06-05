from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import sys
import os
import joblib
import numpy as np

sys.path.append(os.path.abspath(os.path.join("..")))
from src.LSTM.network_variants import LSTMAttentionModel

model = None
scaler = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler
    print("Loading PyTorch model and Scaler into memory...")
    
    scaler_path = "artifacts/trained_model_LSTMAttentionModel_lr0.001:v1/scaler.joblib"
    scaler = joblib.load(scaler_path)
    
    model_path = "artifacts/trained_model_LSTMAttentionModel_lr0.001:v1/model.pth"
    
    model = LSTMAttentionModel(
        input_size=15,
        n_past_features=15,
        n_future_features=4,
        hidden_size=64, 
        num_layers=1, 
        dropout=0.01
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval() 
    
    print("Model and Scaler loaded successfully!")
    yield 
    
    print("Shutting down API and clearing memory...")
    model = None
    scaler = None

app = FastAPI(title="Live Energy LSTM API", lifespan=lifespan)

class PredictionRequest(BaseModel):
    past_sequences: list[list[float]]
    future_features: list[list[float]]

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        Xp = torch.tensor([request.past_sequences], dtype=torch.float32)
        Xf = torch.tensor([request.future_features], dtype=torch.float32)

        if model is not None:
            with torch.no_grad():
                prediction = model(Xp, Xf).view(-1)
                scaled_pred_value = float(prediction.item())
                
                # inverse transform the prediction to get actual kWh
                n_features = scaler.n_features_in_
                preds_full = np.zeros((1, n_features))
                preds_full[0, 0] = scaled_pred_value
                
                real_kwh = scaler.inverse_transform(preds_full)[0, 0]
        else:
            real_kwh = 0.0
            
        return {
            "status": "success",
            "predicted_kWh": round(float(real_kwh), 2)
        }
    except Exception as e:
        print(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))