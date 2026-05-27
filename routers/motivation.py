from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any
import joblib

router = APIRouter()

try:
    motivation_model = joblib.load("models/motivation_pipeline.pkl")
except Exception as e:
    print(f"Warning: Could not load ML model from disk. {e}")
    motivation_model = None

class MotivationRequest(BaseModel):
    user_input: str

class MotivationResponse(BaseModel):
    mental_state: str




@router.post("/motivation", response_model=MotivationResponse)
async def predict_motivation(request: MotivationRequest):
    print("> Received motivation prediction request")

    # Safety check in case the .pkl file is missing from the server
    if not motivation_model:
        raise HTTPException(status_code=500, detail="ML model is not loaded on the server.")

    raw_input = request.user_input
    
    cleaned_input = raw_input.strip()
    
    # If the frontend accidentally sends completely empty text, return Neutral
    if not cleaned_input:
         return MotivationResponse(mental_state="Neutral/Baseline")

    print("> Running text through Scikit-Learn Pipeline...")

    # Predict using the loaded pipeline
    try:
        # The pipeline handles vectorization and classification in one step
        prediction = int(motivation_model.predict([cleaned_input])[0])
    except Exception as e:
        print(f"!! ML Model prediction failed: {e}")
        prediction = 1 # Safe fallback to Neutral

    motivation_mapping = {
        0: "Frustrated/Overwhelmed",
        1: "Neutral/Baseline",
        2: "Highly Motivated"
    }
    
    mental_state = motivation_mapping.get(prediction, "Neutral/Baseline")
    print(f"> Result: {mental_state}")
    
    return MotivationResponse(mental_state=mental_state)
