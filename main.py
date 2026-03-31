from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import date
import model_utils
import uvicorn

app = FastAPI(title="Hospital Patient Volume Forecast API")

# Load model and data once at startup
try:
    model, feature_cols, history_df = model_utils.load_assets()
    print("Model and assets loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model, feature_cols, history_df = None, None, None

# Input structure
class PredictionInput(BaseModel):
    date: date
    shift: str # "Morning" or "Afternoon"
    appointments_booked: int

# Output structure
class PredictionOutput(BaseModel):
    date: str
    shift: str
    appointments_booked: int
    predicted_total_patients: int
    status: str

@app.get("/")
def read_root():
    return {"message": "Hospital Patient Volume Forecast API is active!"}

@app.post("/predict", response_model=PredictionOutput)
def predict_volume(input_data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please run train.py first.")
    
    try:
        # Convert date to string for the utility function
        date_str = input_data.date.strftime("%Y-%m-%d")
        
        # Perform prediction
        result = model_utils.predict(
            model=model,
            history_df=history_df,
            input_date=date_str,
            input_shift=input_data.shift,
            appointments_booked=input_data.appointments_booked,
            feature_cols=feature_cols
        )
        
        return PredictionOutput(
            date=date_str,
            shift=input_data.shift,
            appointments_booked=input_data.appointments_booked,
            predicted_total_patients=result,
            status="Success"
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

import os

if __name__ == "__main__":
    # Render cung cấp cổng qua biến môi trường PORT
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
