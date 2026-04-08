from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="🌥️ Cloud Cost Predictor API")

# Load models
model = joblib.load("cost_predictor.pkl")
features = joblib.load("features.pkl")

class CostRequest(BaseModel):
    vm_count: float
    storage_gb: float
    network_gb: float
    cpu_utilization: float
    hour: int
    day_of_week: int
    month: int
    cost_per_vm: float

@app.get("/")
def root():
    return {"message": "🚀 Cloud Cost Predictor Live!", "status": "ready"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict_cost(request: CostRequest):
    data = [[request.vm_count, request.storage_gb, request.network_gb, 
             request.cpu_utilization, request.hour, request.day_of_week, 
             request.month, request.cost_per_vm]]
    prediction = model.predict(data)[0]
    return {
        "predicted_cost_usd": float(prediction),
        "message": "Cost prediction complete!"
    }

@app.get("/example")
def example():
    data = [[25, 2000, 500, 0.7, 14, 1, 6, 2.5]]
    pred = model.predict(data)[0]
    return {"example_prediction": float(pred)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
